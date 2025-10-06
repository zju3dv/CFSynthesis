import numpy as np
import os
import shutil
import argparse

def process(source_dir, save_dir):
    file_list = np.array(os.listdir(source_dir))

    for file_name in file_list:
        parts = file_name.split('.')
        if len(parts) != 2 or parts[1] != 'png':
            continue
        name_parts = parts[0].split('_')
        
        # 只处理 c01 且最后一位是 0000 的文件
        if name_parts[2] != 'c01' or name_parts[6] != '0000':
            continue
        
        os.makedirs(save_dir, exist_ok=True)

        base_prefix = "_".join(name_parts[0:2])  # e.g., gBR_sFM
        d_num = name_parts[3]
        suffix = "_".join(name_parts[4:6])       # mBR2_ch03

        # 拷贝为 c01-c09 的版本
        for i in range(1, 10):
            c_str = f"c{str(i).zfill(2)}"
            new_name = f"{base_prefix}_{c_str}_{d_num}_{suffix}.png"
            shutil.copyfile(
                os.path.join(source_dir, file_name),
                os.path.join(save_dir, new_name)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    process(args.input_img_path, args.save_path)
