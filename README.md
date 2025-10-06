<!-- # magic-edit.github.io -->
<h1 align='Center'>CFSynthesis: Controllable and Free-view 3D Human Video Synthesis</h1>

<div align='Center'>
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=WhwTrhcAAAAJ"target='_blank'>Liyuan Cui</a><sup>1</sup>&emsp;
    <a href='https://xuxiaogang.com/' target='_blank'>Xiaogang Xu</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=o-cC48AAAAAJ' target='_blank'>Wenqi Dong</a><sup>1</sup>&emsp;
    <a href='https://github.com/YZsZY' target='_blank'>Zesong Yang</a><sup>1</sup>&emsp;
    <a href='https://zhpcui.github.io/' target='_blank'>Zhaopeng Cui</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Hujun Bao</a><sup>1</sup>&emsp;
</div>
<br>
<div align='Center'>
    <sup>1</sup>Zhejiang University<sup>2</sup>The Chinese University of Hong Kong
</div>
<div align='Center'>
<i><strong><a href='https://dl.acm.org/doi/proceedings/10.1145/3731715' target='_blank'>ICMR 2025</a></strong></i>
</div>
<br>
<div align='Center'>
    <a href="https://arxiv.org/abs/2412.11067">
        <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&style=for-the-badge" alt="arXiv">
    </a>
    <a href="https://youtu.be/apJhMlK9yog">
        <img src="https://img.shields.io/badge/Video-Demo-blue?logo=youtube&style=for-the-badge" alt="Video">
    </a>
</div>

# Overeview

![Overeview](assets/teaser.pdf)


# ⚒️ Installation

**Prerequisites:** `python>=3.10`, `CUDA>=11.7`, and `ffmpeg`.

Install dependencies:
- Tested GPUs: A100, We require at least 40 GB of GPU memory.
```bash
pip install -r requirements.txt
```

---

# 🚀 Training and Inference 

## Prepare Datasets

The data processing code is located in `CFSynthesis/render_dataset`.  
Prepare your training data in the following format (we use [ASIT](https://google.github.io/aistplusplus_dataset/factsfigures.html) as an example) in the corresponding folder:

```text
render_dataset/path/to/datasets
  ├── gBR_sFM_c08_d06_mBR5
  │   ├── gBR_sFM_c08_d06_mBR5_0001.png
  │   ├── gBR_sFM_c08_d06_mBR5_0002.png
  │   ...
  ├── gLO_sFM_c01_d13_mLO1
  │   ├── gLO_sFM_c01_d13_mLO1_0001.png
  │   ├── gLO_sFM_c01_d13_mLO1_0002.png
```

We use the following tools (please ensure that all dependencies and pretrained checkpoints are properly set up):

- UV map generation: [SMPLitex](https://github.com/dancasas/SMPLitex)
- Segmentation: [SemanticGuidedHumanMatting](https://github.com/cxgincsu/SemanticGuidedHumanMatting)
- 3D pose estimation: [4D-Humans](https://github.com/shubham-goel/4D-Humans)

Install Detectron2:

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl \
  -P /path/to/detectron2/projects/DensePose/checkpoints/
```

Generate UV maps and background:

```bash
bash process.sh path/to/datasets /absolute/path/to/detectron2
```

Generate foreground:

```bash
# TODO: your command here
```

---

## Inference

Run inference:

```bash
python -m scripts.pipeline.pose2vid \
  --config ./configs/animation/animation.yaml -W 512 -H 512 -L 96
```

---

# 🏋️‍♂️ Training

## Data Preparation

Extract the meta info of your dataset:

```bash
python tools/extract_meta_info.py --root_path /path/to/your/video_dir --dataset_name asit 
```

Update the training config:

```yaml
data:
  meta_paths:
    - "./data/asit_meta.json"
```

---

### Stage 1

Download base models from [Hugging Face](https://huggingface.co/lycui/CFSynthesis).  
We recommend using `git lfs` to download large files.

Place the models as follows:

```text
pretrained_weights
|-- ckpts  
|   |-- denoising_unet.pth
|   |-- guidance_encoder_depth.pth
|   |-- guidance_encoder_dwpose.pth
|   |-- guidance_encoder_normal.pth
|   |-- guidance_encoder_semantic_map.pth
|   |-- reference_unet.pth
|-- control_v11p_sd15_openpose
|   |-- diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

Run Stage 1 training:

```bash
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```

---

### Stage 2

Download the pretrained motion module weights  
[`mm_sd_v15_v2.ckpt`](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)  
and place it under `./pretrained_weights`.

Specify Stage 1 weights in the config file `stage2.yaml`:

```yaml
stage1_ckpt_dir: './exp_output/stage1'
stage1_ckpt_step: 30000 
```

Run Stage 2 training:

```bash
accelerate launch train_stage_2.py --config configs/train/stage2.yaml
```

---

## 🙏 Acknowledgements

This project builds upon the excellent work of:

- [AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)
- [SMPLitex](https://github.com/dancasas/SMPLitex)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [SemanticGuidedHumanMatting](https://github.com/cxgincsu/SemanticGuidedHumanMatting)

We thank the authors for releasing their code and models.

---

## 🎓 Citation

If you find this codebase useful, please cite:

```bibtex
@inproceedings{cui2025cfsynthesis,
  title={CFSynthesis: Controllable and Free-view 3D Human Video Synthesis},
  author={Cui, Liyuan and Xu, Xiaogang and Dong, Wenqi and Yang, Zesong and Bao, Hujun and Cui, Zhaopeng},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={135--144},
  year={2025}
}
```
