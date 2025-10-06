import re
import os
import json
import random

import torch
import torchvision.transforms as transforms
import numpy as np
from decord import VideoReader
from PIL import Image, ImageOps
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
        ####
        self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop( 
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.Resize(
                    self.img_size,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.Resize(
                    self.img_size,
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
        # bg_path = video_meta["bg_path"]
        mask_path = video_meta["mask_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        mask_reader = VideoReader(mask_path)

        # # # change camera 
        # original_id = re.search('c\d{2}', video_path).group()
        # # num = random.choice([i for i in range(1, 10) if i != original_id]) #random.randint(1, 9) #

        # ref_reader = VideoReader(video_path.replace('gt', 'ref').replace(original_id, 'c' + str(1).zfill(2)))

        assert len(video_reader) == len(kps_reader) == len(mask_reader), f"{len(video_reader) = }, {len(kps_reader) = }, {len(mask_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1) #margin) 
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        fg_img = video_reader[ref_img_idx]
        fg_img_pil = Image.fromarray(fg_img.asnumpy())

        fg_mask = Image.fromarray(mask_reader[ref_img_idx].asnumpy()[:, :, 0])
        
        # bg_mask = ImageOps.invert( 
        #     Image.fromarray(mask_reader[tgt_img_idx].asnumpy()[:, :, 0])
        # )
        ###  erosion
        bg_mask = mask_reader[tgt_img_idx].asnumpy()[:, :, 0].astype('uint8')
        bg_mask = ~bg_mask
        kernel = np.ones((5,5),np.uint8)

        import cv2
        dilated_mask = cv2.erode(bg_mask/255, kernel, iterations = 1).astype('uint8')
        # dilated_mask = np.stack([dilated_mask]*3, axis=-1).astype('uint8')

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        tgt_bg_img = self.augmentation(
            Image.composite(tgt_img_pil, Image.new('RGB', tgt_img_pil.size), Image.fromarray(dilated_mask*255)),
           # Image.composite(tgt_img_pil, Image.new('RGB', tgt_img_pil.size), bg_mask), 
            self.cond_transform, 
            state
        )
        ref_img_vae = self.augmentation(
            Image.composite(fg_img_pil, Image.new('RGB', fg_img_pil.size), fg_mask), 
            self.cond_transform, 
            state
        )
        ref_img_mask = self.augmentation(fg_mask, self.cond_transform, state)

        ref_img_mask[ref_img_mask>=0.001] = 1
        ref_img_mask[ref_img_mask<0.001] = 0 # normalize

        #### 随机生成noise
        # coords = (mask == 0).nonzero(as_tuple=True)
        # for coord in zip(*coords):
        #     ref_img_vae[:, coord[0], coord[1]] = torch.rand(3)

        ref_img_vae=torch.concat((ref_img_vae, ref_img_mask),dim=0)

        clip_image = self.clip_image_processor(
            images=Image.composite(fg_img_pil, Image.new('RGB', fg_img_pil.size), fg_mask), return_tensors="pt"
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
