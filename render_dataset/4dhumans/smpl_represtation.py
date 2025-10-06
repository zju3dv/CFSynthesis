import os
os.environ["PYOPENGL_PLATFORM"] =  "osmesa"
import cv2
import json
import torch
import argparse
import pyrender
import trimesh
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm

from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)
# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

def init_mesh(obj_name, device=DEVICE):
    # model_path = os.path.join(args.input_dir, "{}.obj".format(obj_name))
    model_path = obj_name
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)
    return mesh, verts, faces, aux


def projection_pyrender(mesh, image_size, focal, princpt):
    mesh = pyrender.Mesh.from_trimesh(mesh) 
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0)) #0.3, 0.3, 0.3
    scene.add(mesh, 'mesh')
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[1], cy=princpt[0])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb =  Image.fromarray(np.uint8(rgb))
    return rgb
    
class SMPLDataset(Dataset):  
    def __init__(self, folder_path, device, model, model_cfg, save_path, mask_v, mask_faces, uv, renderer, detector, texture, uvsize):
        self.folder_path = folder_path
        self.device = device
        self.names = [f for f in os.listdir(self.folder_path)]
        self.model = model
        self.model_cfg = model_cfg
        self.save_path=save_path
        self.renderer = renderer
        self.detector = detector
        self.texture = texture
        self.mask_v = mask_v
        self.mask_faces = mask_faces
        self.uv = uv
        self.uvsize = uvsize

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        file_name = self.names[idx]
        image_path = os.path.join(self.folder_path, file_name)
        self.pre_mask(image_path)
        return file_name
        
    def pre_mask(self, image_path):

        img_cv2 = cv2.imread(str(image_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        # valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.8)
        max_score_idx = det_instances.scores[det_instances.pred_classes == 0].argmax()
        valid_idx = torch.where(det_instances.pred_classes == 0)[0][max_score_idx]
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()[None, :]
        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
        for batch in dataloader:
            batch = recursive_to(batch, device)
            #### draw box 
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size): # n = 0 forever
                # Get filename from path img_path
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]

                # Save all meshes to disk
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation)#, LIGHT_BLUE)
                
                focal=[scaled_focal_length.item(),scaled_focal_length.item()]
                princpt=[img_size[0][0].item()/2,img_size[0][1].item()/2]
                
                image = self.reproject(tmesh.vertices, focal, princpt)
        
        image.save(os.path.join(self.save_path, os.path.basename(image_path)))
    
    def reproject(self, verts, focal, princpt): 
        vertices = verts[mask_v]
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.mask_faces, process=False) # de-duplicate
        mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, image=self.texture)
        image = projection_pyrender(mesh, self.uvsize, focal, princpt)
        return image
           
if __name__ == "__main__":
    # --------------- Arguments ---------------
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='', help='Folder with input images', required=True)
    parser.add_argument('--out_folder', type=str, default='', help='Output folder to save rendered results',  required=True)
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument("--texture_path", type=str, default=None)
    parser.add_argument("--uv_size", type=int, default=512)
    parser.add_argument("--smpl_uv", type=str, default='', help='path for smpl_uv.obj', required=True)

    args = parser.parse_args()
    _, _, faces, aux = init_mesh(args.smpl_uv)
    faces_verts, uv, faces_uvs = faces.verts_idx, aux.verts_uvs, faces.textures_idx
    mask_faces, mask_v, mask_vt = trimesh.visual.texture.unmerge_faces(faces_verts.cpu().numpy(), faces_uvs.cpu().numpy())
    uv = uv[mask_vt].cpu().numpy()
    
    tetxure = Image.open(args.texture_path).convert("RGB").resize((args.uv_size, args.uv_size))

    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    Segdata = SMPLDataset(args.img_folder, device, model, model_cfg, args.out_folder, mask_v, mask_faces, uv, renderer, detector, tetxure, args.uv_size)

    batch_size = 8 
    data_loader = DataLoader(Segdata, batch_size=batch_size, shuffle=False )
    data_iter = tqdm(data_loader, total=len(data_loader))
    for batch in enumerate(data_iter):
        _ = batch