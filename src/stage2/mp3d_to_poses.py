"""從 MP3D 原生 undistorted_camera_parameters/<scene>.conf 直接建 nerfstudio 格式 poses json。

不依賴 OpenNeRF 未公開的 frame 命名轉換——直接用 MP3D 自帶的相機內外參，
影像路徑指向 MP3D 原生 undistorted 檔（<pano>_i<cam>_<yaw>.jpg / _d<cam>_<yaw>.png）。

MP3D .conf 格式（每行）：
  dataset matterport
  n_images <N>
  depth_directory undistorted_depth_images
  color_directory undistorted_color_images
  intrinsics_matrix fx 0 cx 0 fy cy 0 0 1      # 套用到其後的 scan 行，直到下一個 intrinsics_matrix
  scan <depth_file> <color_file> <16 個 camera-to-world 值，row-major>
  ...

⚠ 座標慣例：.conf 的 camera-to-world 慣例需在第一棟上對照 GT 驗證
   （extract_skeleton 內對 c2w 做 [0:3,1:3]*=-1 的 OpenGL→OpenCV 翻轉；
    若 MP3D 已是 OpenCV 慣例，需用 --no-gl-flip 關閉，見驗證腳本）。
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def parse_conf(conf_path: str, data_root: str):
    """回傳 nerfstudio 風格 dict：{camera_model, frames:[{file_path, depth_file_path,
    fl_x, fl_y, cx, cy, h, w, transform_matrix}]}。"""
    color_dir = "undistorted_color_images"
    depth_dir = "undistorted_depth_images"
    intr = None
    frames = []
    hw_cache = {}
    with open(conf_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            key = tok[0]
            if key == "color_directory":
                color_dir = tok[1]
            elif key == "depth_directory":
                depth_dir = tok[1]
            elif key == "intrinsics_matrix":
                m = list(map(float, tok[1:10]))  # fx 0 cx 0 fy cy 0 0 1
                intr = dict(fx=m[0], cx=m[2], fy=m[4], cy=m[5])
            elif key == "scan":
                depth_file, color_file = tok[1], tok[2]
                ext = np.array(list(map(float, tok[3:19])), dtype=np.float64).reshape(4, 4)
                cpath = os.path.join(data_root, color_dir, color_file)
                dpath = os.path.join(data_root, depth_dir, depth_file)
                # h,w 從影像讀一次（同尺寸多半一致，快取）
                if cpath not in hw_cache:
                    if os.path.exists(cpath):
                        w, h = Image.open(cpath).size
                    else:
                        w, h = None, None
                    hw_cache[cpath] = (h, w)
                h, w = hw_cache[cpath]
                assert intr is not None, f"{conf_path}: scan 行出現在 intrinsics_matrix 之前"
                frames.append(dict(
                    file_path=cpath, depth_file_path=dpath,
                    fl_x=intr["fx"], fl_y=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                    h=h, w=w, transform_matrix=ext.tolist()))
    return {"camera_model": "OPENCV", "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="undistorted_camera_parameters/<scene>.conf")
    ap.add_argument("--data-root", required=True, help="含 undistorted_color_images/ 與 _depth_images/ 的目錄")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    poses = parse_conf(args.conf, args.data_root)
    n = len(poses["frames"])
    miss = sum(1 for fr in poses["frames"]
               if not os.path.exists(fr["file_path"]) or not os.path.exists(fr["depth_file_path"]))
    with open(args.out, "w") as f:
        json.dump(poses, f)
    f0 = poses["frames"][0] if n else {}
    print(f"[mp3d→poses] {n} frames → {args.out}")
    print(f"  首幀 intrinsics: fx={f0.get('fl_x'):.1f} fy={f0.get('fl_y'):.1f} "
          f"cx={f0.get('cx'):.1f} cy={f0.get('cy'):.1f} h={f0.get('h')} w={f0.get('w')}")
    print(f"  缺檔幀數: {miss}/{n}" + ("  ✓" if miss == 0 else "  ⚠ 檢查 data-root/檔名"))


if __name__ == "__main__":
    main()
