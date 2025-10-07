import re
import os
import json
import random

import torch
import torchvision.transforms as transforms
import numpy as np
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class HumanDanceDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        bg_path = video_meta["bg_path"]
        # mask_path = video_meta["mask_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        bg_reader = VideoReader(bg_path)

        assert len(video_reader) == len(kps_reader) == len(bg_reader), f"{len(video_reader) = }, {len(kps_reader) = }, {len(bg_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        '''
        process for asit data
        '''
        match = re.search(r'(g\w+_s\w+_d\w+_m\w+_ch01)', video_path) 
        name = re.sub(re.search(r'c(\d+)', match[0])[0], 'c01', match[0])+'.png'
        path='/'.join(video_path.split('/')[:-1]).replace('train/gt', 'ref')
        ref_path = os.path.join(path, name)
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        fg_mask = Image.fromarray(self.pil2binary_fg(ref_img_pil))

        '''
        process for tiktok data
        '''
        # pattern = r"^(TiktokDance_\d+)(\.mp4)$"
        # match = re.match(pattern,  os.path.basename(video_path))
        # prefix = match.group(1) 
        # path='/'.join(video_path.split('/')[:-1]).replace('gt', 'ref')
        # ref = os.path.join(path, prefix)
        # ref_path = os.path.join(ref, os.listdir(ref)[0])
        
        # ref_img_pil = Image.open(ref_path).convert('RGB')
        # fg_mask = Image.fromarray(self.pil2binary_fg(ref_img_pil))

        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        tgt_bg = bg_reader[tgt_img_idx]
        tgt_bg_pil = Image.fromarray(tgt_bg.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        tgt_bg_img = self.augmentation(tgt_bg_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.cond_transform, state)
        ref_img_mask = self.augmentation(fg_mask, self.cond_transform, state)

        ref_img_mask[ref_img_mask>=0.001] = 1
        ref_img_mask[ref_img_mask<0.001] = 0 # normalize

        #### 随机生成noise
        # coords = (mask == 0).nonzero(as_tuple=True)
        # for coord in zip(*coords):
        #     ref_img_vae[:, coord[0], coord[1]] = torch.rand(3)

        ref_img_vae=torch.concat((ref_img_vae, ref_img_mask),dim=0)

        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            tgt_bg=tgt_bg_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)

    def pil2binary_fg(self, img):
        xx = np.array(img.convert('L'))
        xx[xx > 0] = 255
        xx[xx < 255] = 0
        return xx
