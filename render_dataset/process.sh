# pip install -r requirements.txt
# Download our pretrained diffuser model from SMPLitex-v1.0.zip(https://drive.google.com/file/d/1vLLxknRjvQU1oqYha749EYpLPoK1Jn7U/view?usp=sharing) and unzip it ./simplitex-trained-model
# Download Detectron2 (https://github.com/facebookresearch/detectron2), python -m pip install -e detectron2 
# Download weights and put it in  SemanticGuidedHumanMatting/pretrain.

folder=$1
detectron2=$2

for path in "$folder"/*/; do    
    if [ -d "$path" ]; then
        images_dir="${path}/images" 

        mkdir -p "${images_dir}"
        mv "${path}"/*.jpg "${images_dir}" 2>/dev/null || true
        mv "${path}"/*.png "${images_dir}" 2>/dev/null || true

        current_dir=$(cd "$(dirname "$0")" && pwd)
        absolute_path="${current_dir}/${detectron2}"
        python scripts/image_to_densepose.py --input_folder  ${images_dir} --detectron2 ${detectron2}

        save_path=$(echo $images_dir | sed 's/images/images-seg/')
        python SemanticGuidedHumanMatting/test_image.py --images_dir ${images_dir}  --result_dir  ${save_path} --pretrained_weight SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth

        texture_path=$(dirname "$images_dir")
        python scripts/compute_partial_texturemap.py --input_folder ${texture_path}

        character_name=$(basename $path)
        python scripts/inpaint.py  --guidance_scale 2.5 --inference_steps 250 --model_path simplitex-trained-model --output_dir texture_map --image_path ${character_name}.png --mask_path ${path}/uv-textures-masks/images_mask.png
        
        bg_path="$(dirname "$path")/ref_control" #$(echo $images_dir | sed 's/images/ref_control/')
        python SemanticGuidedHumanMatting/seg_bg_image_folder.py --images_dir ${images_dir}  --result_dir ${bg_path} --pretrained_weight SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth

        cond_path="$(dirname "$path")/cond" #$(echo $images_dir | sed 's/images/cond/')
        python  4dhumans/smpl_represtation.py --img_folder ${images_dir} --out_folder ${cond_path} --texture_path texture_map/${character_name}.png --smpl_uv 4dhumans/smpl_uv.obj

        gt_path="$(dirname "$path")/gt"
        mkdir -p "${gt_path}"
        mv "${path}"/*.jpg "${gt_path}" 2>/dev/null || true
        mv "${path}"/*.png "${gt_path}" 2>/dev/null || true
    fi
done 
find ${folder} -mindepth 1 -maxdepth 1 ! -name 'ref_control' ! -name 'cond' ! -name 'gt' -exec rm -rf {} +
