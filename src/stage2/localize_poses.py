"""把 HF poses json 內作者機器的絕對路徑改寫成本機 MP3D→nerfstudio 資料路徑。

HF poses（OpenNeRF/nerfstudio 格式）的 file_path/depth_file_path 指向作者機器
（/mnt/usb_ssd/bieriv/opennerf-data/nerfstudio/matterport_<scene>/...），本機不存在。
本工具依 <data-root> 重寫為 <data-root>/images/<stem>.jpg 與 <data-root>/depths/<stem>.png，
並驗證檔案存在（缺檔即報表），輸出 localized json 供 extract_skeleton / openseg 使用。
"""
import argparse
import json
import os


def localize(poses_file: str, data_root: str, out_file: str,
             img_sub: str = "images", depth_sub: str = "depths",
             img_ext: str = ".jpg", depth_ext: str = ".png") -> None:
    with open(poses_file) as f:
        poses = json.load(f)
    miss_img = miss_depth = 0
    for fr in poses["frames"]:
        stem = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        ip = os.path.join(data_root, img_sub, stem + img_ext)
        dp = os.path.join(data_root, depth_sub, stem + depth_ext)
        fr["file_path"] = ip
        fr["depth_file_path"] = dp
        miss_img += not os.path.exists(ip)
        miss_depth += not os.path.exists(dp)
    with open(out_file, "w") as f:
        json.dump(poses, f)
    n = len(poses["frames"])
    print(f"[localize] {n} frames → {out_file}")
    if miss_img or miss_depth:
        print(f"  ⚠ 缺檔: images {miss_img}/{n}, depths {miss_depth}/{n} "
              f"（檢查 --data-root 與 img/depth 子目錄命名）")
    else:
        print(f"  ✓ 全部 {n} 幀影像＋深度檔存在")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-file", required=True)
    ap.add_argument("--data-root", required=True, help="本機 nerfstudio 場景根（含 images/ depths/）")
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--img-ext", default=".jpg")
    ap.add_argument("--depth-ext", default=".png")
    a = ap.parse_args()
    localize(a.poses_file, a.data_root, a.out_file, img_ext=a.img_ext, depth_ext=a.depth_ext)
