from UVTextureConverter import UVConverter
from UVTextureConverter import Atlas2Normal
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import skimage
import os

class RGB2Texture:

    def __init__(self, dataset_root_path) -> None:
        self.dataset_root_path = dataset_root_path
        
    def apply_mask_to_iuv_images(self):
        dataset_iuv_path = os.path.join(self.dataset_root_path, 'densepose')
        dataset_mask_path = os.path.join(self.dataset_root_path, 'images-seg')

        if not os.path.exists(dataset_iuv_path):
            print("ERROR: ", dataset_iuv_path, " does not exist")
            return

        if not os.path.exists(dataset_mask_path):
            print("ERROR: ", dataset_mask_path, " does not exist")
            return

        # output path for masked iuv
        output_iuv_masked_folder = "densepose-masked"
        output_iuv_masked_folder_path = os.path.join(self.dataset_root_path, output_iuv_masked_folder)
        isExist = os.path.exists(output_iuv_masked_folder_path)
        if not isExist:
            os.makedirs(output_iuv_masked_folder_path)

        files = os.listdir(dataset_iuv_path)

        num_images = 0

        # reads all the images and stores full paths in list
        for path in files:

            num_images += 1
            current_iuv_path = os.path.join(dataset_iuv_path, path)
            current_mask_path = path.replace("_densepose.png", ".png")
            current_mask_path = os.path.join(dataset_mask_path, current_mask_path)

            if os.path.isfile(current_iuv_path):
                if os.path.isfile(current_mask_path):

                    with Image.open(current_iuv_path) as im_iuv:
                        with Image.open(current_mask_path) as im_mask:

                            print('\nSegmenting image ', num_images, '/', len(files))
                            #print('Loading      ', current_iuv_path)
                            #print('Loading      ', current_mask_path)

                            iuv_w, iuv_h = im_iuv.size
                            mask_w, mask_h = im_mask.size

                            if (iuv_w == mask_w) and (iuv_h == mask_h):
                                
                                threshold = 250
                                im_mask = im_mask.point(lambda x: 255 if x > threshold else 0)
                                blank = im_iuv.point(lambda _: 0)
                                masked_iuv_image = Image.composite(im_iuv, blank, im_mask)
                                
                                print('Writing image ', os.path.join(output_iuv_masked_folder_path, path))
                                masked_iuv_image.save(os.path.join(output_iuv_masked_folder_path, path), "PNG")
                            else:

                                print('Discarding images because densepose and RGB do not match. Probably densepose failed?')
                                
            
                else:
                    print(current_mask_path, 'does not exist')
            else:
                print(current_iuv_path, 'does not exist')


    def custom_median_filter(self, texmap_list):
        H, W, C = texmap_list[0].shape
        complete_map = np.zeros((H, W, C), dtype=texmap_list[0].dtype)

        for h in range(H):
            for w in range(W):
                for c in range(C):                    
                    pixel_values = np.array([texmap[h, w, c] for texmap in texmap_list])
                    non_zero_values = pixel_values[pixel_values != 0]
                    
                    if non_zero_values.size > 0:
                        complete_map[h, w, c] = np.median(non_zero_values)
                        
        return complete_map

    def generate_uv_texture(self):
    
        # paths
        dataset_image_path = os.path.join(self.dataset_root_path, 'images')
        dataset_iuv_path = os.path.join(self.dataset_root_path, 'densepose-masked')
        
        # output path for UV textures
        output_textures_folder = "uv-textures"
        output_textures_folder_path = os.path.join(self.dataset_root_path, output_textures_folder)
        isExist = os.path.exists(output_textures_folder_path)
        if not isExist:
            os.makedirs(output_textures_folder_path)

        # output path for debug figure
        output_debug_folder = "debug"
        output_debug_folder_path = os.path.join(self.dataset_root_path, output_debug_folder)
        isExist = os.path.exists(output_debug_folder_path)
        if not isExist:
            os.makedirs(output_debug_folder_path)

        # list to store files
        images_file_paths = []
        images_iuv_paths = []

        num_images = 0

        # extract list of image files
        files = os.listdir(dataset_image_path)

        # size (in pixels) of each part in texture
        # WARNING: for best results, update this value depending on the input image size
        parts_size = 120 

        # reads all the images and stores full paths in list
        for path in files:
            num_images += 1
            current_image_path = os.path.join(dataset_image_path, path)
            current_iuv_path = os.path.join(dataset_iuv_path, path.replace(".png", "_densepose.png"))

            # check if both image and iuv exists
            if os.path.isfile(current_image_path):
                if os.path.isfile(current_iuv_path):
                    images_file_paths.append(current_image_path)
                    images_iuv_paths.append(current_iuv_path)

                else:
                    print(current_iuv_path, ' does not exist')
            else:
                print(current_image_path, ' does not exist')

        num_images = 0

        # sorts filenames alphabetically
        images_iuv_paths.sort()
        images_file_paths.sort()

        images_iuv_paths_filtered = images_iuv_paths.copy()
        images_file_paths_filtered = images_file_paths.copy()
        num_images = 0

        # normal_tex_pre = np.zeros((512, 512, 3))
        texmap_list = []
        
        for current_image_path, current_iuv_path in tqdm(zip(images_file_paths_filtered, images_iuv_paths_filtered), total=len(images_file_paths_filtered)):

            num_images += 1
            tex_trans, mask_trans = UVConverter.create_texture(current_image_path, current_iuv_path,
            parts_size=parts_size, concat=False)
            texture_size = 512 

            converter = Atlas2Normal(atlas_size=parts_size, normal_size=texture_size)
            normal_tex, normal_ex = converter.convert((tex_trans*255).astype('int'), mask=mask_trans)
            
            texmap_list.append(normal_tex)

            # nonzero_indices = normal_tex > 0
            # normal_tex_pre[nonzero_indices] = normal_t ex[nonzero_indices]

        # texmap_list = np.stack(texmap_list)
        complete_map = self.custom_median_filter(texmap_list)  ##np.median(texmap_list, axis=0)

        output_textures_file_path = os.path.join(output_textures_folder_path, os.path.basename(dataset_image_path)+'.png')
        #normal_tex = (normal_tex_pre * 255).round().astype(np.uint8)
        normal_tex = (complete_map * 255).round().astype(np.uint8)
        im = Image.fromarray(normal_tex, 'RGB')
        im.save(output_textures_file_path)

    def compute_mask_of_partial_uv_textures(self):

        # folder where the uv textures are. Ideally, these are textures generated with masked images 
        # (e.g, the IUV images where segmented using masks computed on the rendered images)
        uv_textures_path = os.path.join(self.dataset_root_path, 'uv-textures')
        
        files = os.listdir(uv_textures_path)

        # output path for UV masks
        output_masks_folder = "uv-textures-masks"
        output_masks_folder_path = os.path.join(self.dataset_root_path, output_masks_folder)
        isExist = os.path.exists(output_masks_folder_path)
        if not isExist:
            os.makedirs(output_masks_folder_path)

        num_images = 0

        # reads all the images and computes UV mask
        for image_filename in files:
            num_images += 1
            print('\nComputing UV mask ', num_images, '/', len(files))

            current_image = skimage.io.imread(os.path.join(uv_textures_path, image_filename))

            mask = (current_image[:, :, 0] < 2) & (current_image[:, :, 1] < 2) & (current_image[:, :, 1] < 2)

            mask_filename = image_filename.replace(".png", "_mask.png")
            print('Writing image ', os.path.join(output_masks_folder_path, mask_filename))
            skimage.io.imsave(os.path.join(output_masks_folder_path, mask_filename), skimage.img_as_ubyte(mask))

parser = argparse.ArgumentParser(description= 'Computes partial texturemap from RGB image')
parser.add_argument('--input_folder', type=str, help='Root folder with subfolders images/ and densepose/', required=True)

args = parser.parse_args()

INPUT_FOLDER = args.input_folder

# removes backslash in case it's the last character of the input path
if INPUT_FOLDER.endswith('/'):
    INPUT_FOLDER = INPUT_FOLDER[:-1]

densepose = RGB2Texture(INPUT_FOLDER)
densepose.apply_mask_to_iuv_images()
densepose.generate_uv_texture()
densepose.compute_mask_of_partial_uv_textures()