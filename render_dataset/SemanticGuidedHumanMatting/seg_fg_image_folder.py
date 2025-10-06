import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.model import HumanSegment, HumanMatting
import utils
import inference
class SegDataset(Dataset):  
    def __init__(self, folder_path, device, model, save_path):
        self.folder_path = folder_path
        self.device = device
        self.names = [f for f in os.listdir(self.folder_path)]
        self.model = model
        self.save_path=save_path

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        file_name = self.names[idx]
        image_path = os.path.join(self.folder_path, file_name)
        self.pre_mask(image_path)
        return file_name
        
    
    def pre_mask(self, image_path):
        image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
        pred_alpha, pred_mask = inference.single_inference(model, img)

        Img = np.array(img)

        background = (pred_alpha * 255).astype('uint8')
        background[background < 80] = 0
        background[background >= 80] = 1
        background_3d = np.expand_dims(background, axis=-1)
        Img = Img * background_3d

        # save results
        output_dir = os.path.join(args.result_dir,"_".join(image_name.split('_')[:2]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path=os.path.join(output_dir, image_name + '.png')
        Image.fromarray(Img, 'RGB').save(save_path)

    
if __name__ == "__main__":
    # --------------- Arguments ---------------
    parser = argparse.ArgumentParser(description='Test Images')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--pretrained_weight', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.pretrained_weight):
        print('Cannot find the pretrained model: {0}'.format(args.pretrained_weight))
        exit()

    # Load Model
    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).cuda().eval()
    model.load_state_dict(torch.load(args.pretrained_weight))
    print("Load checkpoint successfully ...")

    os.makedirs(args.result_dir, exist_ok=True)
    Segdata = SegDataset(args.images_dir, 0, model, args.result_dir)
    batch_size = 64 #len(pose_data.file_names)
    data_loader = DataLoader(Segdata, batch_size=batch_size, shuffle=False)
    data_iter = tqdm(data_loader, total=len(data_loader))
    for batch in enumerate(data_iter):
        _ = batch