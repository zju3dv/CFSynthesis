import os
import re
import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
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
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        bg_path = video_meta["bg_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        bg_reader = VideoReader(bg_path)

        assert len(video_reader) == len(kps_reader) == len(bg_reader), f"{len(video_reader) = }, {len(kps_reader) = }, {len(bg_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # match = re.search(r'(g\w+_s\w+_d\w+_m\w+_ch01)', video_path) ### for asit
        # name = re.sub(re.search(r'c(\d+)', match[0])[0], 'c01', match[0])+'.png'
        # path='/'.join(video_path.split('/')[:-1]).replace('train/gt', 'ref')
        # ref_path = os.path.join(path, name)
        
        pattern = r"^(TiktokDance_\d+)(\.mp4)$"
        match = re.match(pattern,  os.path.basename(video_path))
        prefix = match.group(1) 
        path='/'.join(video_path.split('/')[:-1]).replace('gt', 'ref')
        ref = os.path.join(path, prefix)
        ref_path = os.path.join(ref, os.listdir(ref)[0])
        ref_img = Image.open(ref_path).convert('RGB')
        fg_mask = Image.fromarray(self.pil2binary_fg(ref_img))

        # # # change camera 
        # original_id = re.search('c\d{2}', video_path).group()
        # # num = random.choice([i for i in range(1, 10) if i != original_id]) #random.randint(1, 9) #

        # ref_reader = VideoReader(video_path.replace('gt', 'ref').replace(original_id, 'c' + str(1).zfill(2)))

        # ref_idx = random.randint(0, len(ref_reader) - 1)
        # ref_img = Image.fromarray(ref_reader[ref_idx].asnumpy())
        # fg_mask = Image.fromarray(self.pil2binary_fg(ref_img))

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        bg_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = bg_reader[index]
            bg_pil_image_list.append(Image.fromarray(img.asnumpy()))
        
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_bg = self.augmentation(
            bg_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.cond_transform, state)
        pixel_values_ref_mask = self.augmentation(fg_mask, self.cond_transform, state)
        pixel_values_ref_img=torch.concat((pixel_values_ref_img, pixel_values_ref_mask),dim=0)

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_bg=pixel_values_bg,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)

    def pil2binary_fg(self, img):
        xx = np.array(img.convert('L'))
        xx[xx > 0] = 255
        xx[xx < 255] = 0
        return xx