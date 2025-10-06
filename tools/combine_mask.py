import os
import re
import argparse
import numpy as np
from PIL import Image, ImageOps

"""
Construct misaligned backgrounds to enhance the learning capability of the background encoder.
"""

def cyclic_increment(match):
    number = int(match.group(0)[1:])  # 
    new_number = number% 9  + 1 if number != 9 else number% 9  + 2
    return 'c' + str(new_number).zfill(2)  

def combine_mask(name, input_dir, save_dir):
    mask1 = os.path.join(input_dir, 'masks', name)
    mask2 = os.path.join(re.sub(r'c\d+', cyclic_increment, input_dir), 'masks', re.sub(r'c\d+', cyclic_increment, name))
    img = os.path.join(input_dir, 'images', name)

    img = np.asarray(Image.open(img))
    img1 =  ImageOps.invert(Image.open(mask1)).convert('L')
    try:
        img2 = ImageOps.invert(Image.open(mask2)).convert('L')
    except:
        mask2=mask1
        img2 = ImageOps.invert(Image.open(mask2)).convert('L')

    img1_np = np.asarray(img1)/ 255
    img2_np = np.asarray(img2)/255

    result_np = img1_np * img2_np
    result_np = np.expand_dims(result_np, axis=-1)
    img = img*result_np

    result_img = Image.fromarray(img.astype('uint8'))
    result_img.save(os.path.join(save_dir, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Video')
    parser.add_argument('--video_name', default=None, type=str, help='image frame path')
    parser.add_argument('--save_dir', default=None, type=str, help='save bg image after mask')
    args = parser.parse_args()

    images = [img for img in os.listdir(os.path.join(args.video_name, 'images')) if img.endswith(".png")]

    save_dir = os.path.join(args.save_dir, os.path.basename(args.video_name))
    os.makedirs(save_dir, exist_ok=True)

    for file in images:
        combine_mask(file, args.video_name, save_dir)

