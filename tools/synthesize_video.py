import os
import argparse
import imageio.v2 as imageio
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="dataset root path")
parser.add_argument("--fps", type=int, default=25, help="video frame rate")
parser.add_argument("--clean", action="store_true", help="delete PNG frames after saving video")
args = parser.parse_args()

folders = ["gt", "ref_control", "cond", "images-seg"]

for folder in folders:
    folder_path = os.path.join(args.root, folder)
    if not os.path.exists(folder_path):
        continue

    prefixes = sorted(
        set(["_".join(f.split("_")[:-1]) for f in os.listdir(folder_path) if f.endswith(".png")])
    )

    for prefix in prefixes:
        frames = natsorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.startswith(prefix) and f.endswith(".png")
        ])
        if not frames:
            continue

        out_path = os.path.join(folder_path, f"{prefix}.mp4")
        if os.path.exists(out_path):
            print(f"⏩ skip: {out_path}")
            continue

        imageio.mimwrite(
            out_path,
            [imageio.imread(f) for f in frames],
            fps=args.fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        print(f"✅ saved: {out_path}")

        if args.clean:
            for f in frames:
                os.remove(f)
            print(f"🧹 deleted {len(frames)} PNG frames for {prefix}")