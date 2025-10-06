import argparse
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def init_mesh(obj_name):
    model_path = obj_name
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)
    return mesh, verts, faces, aux

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="simplitex-trained-model",
        type=str,
        help="Path to the model to use.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=2, help="Value of guidance step"
    )

    parser.add_argument(
        "--guidance_scale_refinement",
        type=float,
        default=1,
        help="Value of guidance step for refining steps. Not used when --refine is False.",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=200, help="Number of inference steps"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a sks texturemap",
        help="Prompt to use. Use sks texture map as part of your prompt for best results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Folder onto which to save the results.",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default="data_inpainting/mask_example.png",
        help="Path to mask image.",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="data_inpainting/input_example.png",
        help="Path to input image.",
    )

    parser.add_argument(
        "--refine",
        type=bool,
        default=False,
        help="Set to True if you want to refine the results after inpainting",
    )

    parser.add_argument(
        "--render",
        type=bool,
        default=False,
        help="Set to True if you want to render the results",
    )

    args = parser.parse_args()

    assert args.guidance_scale >= 0.0, "Invalid guidance scale value"
    assert args.inference_steps > 0, "Invalid inference steps number"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path, safety_checker=None
    )
    pipe.to("cuda")

    image = load_image(args.image_path)
    mask_image = load_image(args.mask_path)

    image = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.inference_steps,
        strength=1,
    ).images[0]

    if args.refine:
        assert args.guidance_scale_refinement >= 0.0, "Invalid guidance scale value"
        print("Refining in two steps")
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_path, safety_checker=None
        )
        pipe.to("cuda")
        image = pipe(
            prompt=args.prompt,
            image=image,
            guidance_scale=args.guidance_scale_refinement,
            num_inference_steps=args.inference_steps * 4,
            strength=0.05,
        ).images[0]

        image = pipe(
            prompt=args.prompt,
            image=image,
            guidance_scale=args.guidance_scale_refinement,
            num_inference_steps=args.inference_steps * 20,
            strength=0.01,
        ).images[0]

    # path_save = os.path.join(os.getcwd(), args.output_folder)

    # if not os.path.exists(path_save):
    #     os.mkdir(path_save)
    name = os.path.basename(args.output_dir)
    os.makedirs(os.path.basename(args.output_dir), exist_ok=True)   
    image.save(os.path.join(args.output_dir, name+".png"))