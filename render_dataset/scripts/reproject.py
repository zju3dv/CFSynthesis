import os
os.environ["PYOPENGL_PLATFORM"] =  "osmesa"
import matplotlib.pyplot as plt
import torch
import pyrender
import trimesh
import json
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torchgeometry as tgm
from PIL import Image
from torchvision import transforms
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes,
)
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesUV,
    look_at_view_transform
)
from tqdm import tqdm
from pytorch3d.transforms import Transform3d

import sys
sys.path.append(".")
# from lib.camera_helper import init_camera
from lib.render_helper import init_renderer
from lib.shading_helper import (
    BlendParams,
    init_soft_phong_shader, 
)
from lib.load_yaml_data import load_yaml_data
# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

def init_mesh(obj_name, device=DEVICE):
    print("=> loading target mesh...")
    # model_path = os.path.join(args.input_dir, "{}.obj".format(obj_name))
    model_path = obj_name
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)
    return mesh, verts, faces, aux

def normalization(mesh):
    bbox = mesh.get_bounding_boxes()
    num_verts = mesh.verts_packed().shape[0]

    # move mesh to origin
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = mesh.offset_verts(-mesh_center)

    # scale
    lens = bbox[0, :, 1] - bbox[0, :, 0]
    max_len = lens.max()
    scale = 1 / max_len
    scale = scale.unsqueeze(0).repeat(num_verts)
    mesh.scale_verts_(scale)
    return  mesh #mesh.verts_packed()

def normalization_trimesh(mesh):
    bbox = mesh.bounding_box.bounds
    num_verts = len(mesh.vertices)
    mesh_center = np.mean(bbox, axis=0)
    mesh.vertices -= mesh_center
    lens = bbox[:, 1] - bbox[:, 0]
    max_len = np.max(lens)
    scale = 1 / max_len
    mesh.vertices *= scale
    return mesh

def init_camera(R, T, focal,image_size, device):
# def init_camera(dist, elev, azim, image_size, device):
    # R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor(image_size).unsqueeze(0) #torch.tensor([image_size, image_size]).unsqueeze(0)
    prp_screen = (image_size/2).detach().clone()
    cameras = PerspectiveCameras(focal_length=(focal,), R=R, T=T, device=device, image_size=image_size, principal_point=prp_screen, in_ndc=False,) #[12139.43640391032, 12139.437198638916]
    return cameras

# def reprojection(mesh, image_size, faces_per_pixel, device, extri_yaml, align_T, azim, elev, dist): 
def reprojection(mesh, image_size, faces_per_pixel, device, extri_yaml, align_T, pose): 
    # render the view
    verts = torch.cat([mesh.verts_packed(), torch.ones_like(mesh.verts_packed()[:, :1])], dim=-1)
    align_T = torch.tensor(align_T).to(device).to(dtype=verts.dtype)
    transformed_verts = torch.matmul(verts, align_T.t())
    transformed_vertices = transformed_verts[:, :3] #/ 
    transformed_verts[:, 3:]
    
    # change y, z coor ? 这个地方不对
    new_vertices = transformed_vertices.clone()
    # new_vertices[:, 1], new_vertices[:, 2] = transformed_vertices[:, 2], transformed_vertices[:, 1]
    new_vertices[:, 2] *= -1
    mesh.verts_padded()[:] = new_vertices

    # camera_pose = torch.tensor(camera_pose).to(device).to(dtype=verts.dtype)

    R, T = pose[:3, :3][None,:], pose[:3, 3][None,:]
    # azimuth, elevation = get_angle(camera_pose)
    # dist=6.0
    # elev=15.0
    cameras = init_camera(
        R, T, #dist, elev, azim, # 
        35,
        image_size, device
    )
    renderer = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )
    images, fragments = renderer(mesh)
    inter_image = images[0]. permute(2, 0, 1)
    inter_image = transforms.ToPILImage()(inter_image).convert("RGB")

    depth_maps_tensor = get_relative_depth_map(fragments)
    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    return inter_image, depth_map

def projection_pyrender_trace(mesh, image_size, extri_yaml=None, align_pose=None):
    camera_pose, focal, princpt = get_scene_info(extri_yaml)
    if align_pose:
        mesh.apply_transform(align_pose)
    # mesh.apply_transform(camera_pose)
    # rot = trimesh.transformations.rotation_matrix(
    # np.radians(180), [1, 0, 0])
    # mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0)) #0.3, 0.3, 0.3
    scene.add(mesh, 'mesh')
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[1], cy=princpt[0])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size[1], viewport_height=image_size[0], point_size=1.0)

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
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb, depth = Image.fromarray(np.uint8(rgb)), Image.fromarray(np.uint8(depth))
    return rgb, depth

def projection_pyrender(mesh, image_size, meta):
    rot = trimesh.transformations.rotation_matrix(
    np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    with open(meta, 'r') as f:
        data = json.load(f)
        focal = data['focal']
        princpt = data['princpt']

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
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb, depth = Image.fromarray(np.uint8(rgb)), Image.fromarray(np.uint8(depth))
    return rgb, depth

def get_angle(pose):
    # with open(pose_path, 'r') as file:
    #     pose = np.array(json.load(file)['cam2world_n'])
    rotation_matrix = pose[:3, :3]
    elevation = np.arcsin(-rotation_matrix[1, 2])
    azimuth = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)
    return azimuth, elevation

def get_relative_depth_map(fragments, pad_value=10):
    absolute_depth = fragments.zbuf[..., 0] # B, H, W
    no_depth = -1

    depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
    target_min, target_max = 50, 255

    depth_value = absolute_depth[absolute_depth != no_depth]
    depth_value = depth_max - depth_value # reverse values

    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min

    relative_depth = absolute_depth.clone()
    relative_depth[absolute_depth != no_depth] = depth_value
    relative_depth[absolute_depth == no_depth] = pad_value # not completely black
    return relative_depth

def rotation_vertices(vertices):
    angle_x = torch.tensor([180.0, 0, 0])       
    angle_y = torch.tensor([0, 180.0, 0])
    angle_rad_x = torch.deg2rad(angle_x)
    angle_rad_y = torch.deg2rad(angle_y)

    rotation_matrix_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(angle_rad_x[0]), -torch.sin(angle_rad_x[0])],
                        [0, torch.sin(angle_rad_x[0]), torch.cos(angle_rad_x[0])]]).cuda()
    rotation_matrix_y = torch.tensor([[torch.cos(angle_rad_y[1]), 0, torch.sin(angle_rad_y[1])],
                                  [0, 1, 0],
                                  [-torch.sin(angle_rad_y[1]), 0, torch.cos(angle_rad_y[1])]]).cuda()
    rotated_vertices = vertices.mm(rotation_matrix_y.t()) #.mm(rotation_matrix_x.t())
    return rotated_vertices

def angle_from_blender(filename):
    data = np.loadtxt(filename, delimiter=',')
    frames = data[:, 0]
    locations = data[:, 1:4]
    rotations = data[:, 4:7]

    distances = np.sqrt(np.sum(locations**2, axis=-1))

    azimuths = np.arctan2(locations[:,1], locations[:,0])
    elevations = np.arctan2(locations[:,2], np.sqrt(locations[:,0]**2 + locations[:,1]**2))
    azimuths = np.rad2deg(azimuths)
    elevations = np.rad2deg(elevations)
    return azimuths, elevations, distances
    

def get_scene_info(yaml_path):
    extri = load_yaml_data(yaml_path)
    intri = load_yaml_data(yaml_path.replace("extri", "intri"))
    name = "00" 
    translation = extri[f"T_{name}"]
    rotation_matrix = extri[f"Rot_{name}"]
    extrinsics_matrix = np.concatenate((rotation_matrix, translation.reshape(-1, 1)), axis=1)
    extrinsics_matrix =  np.vstack((extrinsics_matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    extrinsics_matrix =  np.linalg.inv(extrinsics_matrix)#[:3, :]
    focal = [intri[f"K_{name}"][0][0], intri[f"K_{name}"][1][1]]
    princpt = [intri[f"K_{name}"][1][2], intri[f"K_{name}"][0][2]]
    return extrinsics_matrix, focal, princpt 

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_texture_path", type=str, default=None)
    parser.add_argument("--input_mesh_path", type=str,default=None)
    parser.add_argument("--save_image_path", type=str, default=None)
    parser.add_argument("--folder_smpl_path", type=str, default=None)
    parser.add_argument("--depth_map_path", type=str, default=None)
    parser.add_argument("--align_pose_path", type=str, default=None)
    parser.add_argument("--cam_info_path", type=str, default=None)
    parser.add_argument("--render_method", type=str, default='pytorch3D')
    parser.add_argument("--uv_size", type=int, default=512)
    # parser.add_argument("--image_size", nargs='+', action='append', type=list, default=[512, 512])
    parser.add_argument("--folder", type=str_to_bool,
                            nargs='?', const=True, default=False)
    args = parser.parse_args()

    setattr(args, "fragment_k", 1)
    os.makedirs(args.save_image_path, exist_ok=True)
    if args.align_pose_path:
        with open(args.align_pose_path, 'r') as file:
            data = json.load(file)
            align_pose = np.array(data['cam2world'])
    if args.render_method == "pytorch3D":
        if args.folder:
            mesh_list = [os.path.join(args.folder_smpl_path, f) for f in os.listdir(args.folder_smpl_path)]
            init_texture = Image.open(args.input_texture_path).convert("RGB").resize((args.uv_size, args.uv_size))
            azim, elev, dist = angle_from_blender('output.txt')
            poses = pose_from_blender('output.txt')
            for mesh_id in tqdm(range(len(azim))):
                mesh_path = mesh_list[mesh_id]
                mesh, _, faces, aux = init_mesh(mesh_path)   
                # update the mesh
                mesh.textures = TexturesUV(
                    maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                    faces_uvs=faces.textures_idx[None, ...],
                    verts_uvs=aux.verts_uvs[None, ...]
                )
                # azim = mesh_id-90
                image, _ = reprojection(mesh, [512,512], args.fragment_k, DEVICE, args.cam_info_path, align_pose,poses[mesh_id])# azim[mesh_id],elev[mesh_id], dist[mesh_id])
                image.save(os.path.join(args.save_image_path , os.path.basename(mesh_path).replace(".obj", "")))

        mesh, _, faces, aux = init_mesh(args.input_mesh_path)   
        init_texture = Image.open(args.input_texture_path).convert("RGB").resize((args.uv_size, args.uv_size))
        # update the mesh
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=aux.verts_uvs[None, ...]
        )
        image, _ = reprojection(mesh, [512,512], args.fragment_k, DEVICE, args.cam_info_path, align_pose)

    if args.render_method == "pyrender":
        if args.folder:
            init_texture = Image.open(args.input_texture_path).convert("RGB").resize((args.uv_size, args.uv_size))
            mesh_list = [os.path.join(args.folder_smpl_path, f) for f in os.listdir(args.folder_smpl_path)]
            for mesh_id in tqdm(range(len(mesh_list))):
                mesh_path = mesh_list[mesh_id]
                _, verts, faces, aux = init_mesh(mesh_path)   
                faces_verts, uv, faces_uvs = faces.verts_idx, aux.verts_uvs, faces.textures_idx
                mask_faces, mask_v, mask_vt = trimesh.visual.texture.unmerge_faces(faces_verts.cpu().numpy(), faces_uvs.cpu().numpy())
                vertices = verts[mask_v].cpu().numpy()
                uv = uv[mask_vt].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vertices, faces=mask_faces, process=False) # de-duplicate
                mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=init_texture)
                image, _ = projection_pyrender(mesh, args.uv_size, mesh_path.replace("mesh", "meta").replace("obj", "json"))
                image.save(os.path.join(args.save_image_path , os.path.basename(mesh_path).replace(".obj", "")))
        else:
            _, verts, faces, aux = init_mesh(args.input_mesh_path)   
            faces_verts, uv, faces_uvs = faces.verts_idx, aux.verts_uvs, faces.textures_idx
            mask_faces, mask_v, mask_vt = trimesh.visual.texture.unmerge_faces(faces_verts.cpu().numpy(), faces_uvs.cpu().numpy())
            vertices = verts[mask_v].cpu().numpy()
            uv = uv[mask_vt].cpu().numpy()
            # vertices= align(args.align_pose_path,  vertices)
            mesh = trimesh.Trimesh(vertices=vertices, faces=mask_faces, process=False) # de-duplicate
            init_texture = Image.open(args.input_texture_path).convert("RGB").resize((args.uv_size, args.uv_size))
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=init_texture)
            image, _ = projection_pyrender(mesh, [1080,1920], args.cam_info_path, align_pose)
            image.save(os.path.join(args.save_image_path , os.path.basename(args.input_mesh_path).replace(".obj", "")))