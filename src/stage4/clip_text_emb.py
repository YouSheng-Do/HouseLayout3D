"""D.4 房型文字嵌入（oneformer env）：附錄的 15 類 → CLIP ViT-L/14@336 text encoder → text_emb.npy。
最後五類為室外類（剪枝用）。"""
import argparse

import numpy as np
import torch

ROOM_TYPES = [
    "bathroom", "bedroom", "living room", "garage", "entrance",
    "kitchen", "office", "stairs", "gym", "classroom",
    "spa/sauna", "mirror", "grass/bushes/trees", "driveway", "veranda/terrace/balcony",
]
N_OUTDOOR_TAIL = 5  # 最後五類＝室外，D.4 剪枝


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    args = ap.parse_args()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(args.model)
    proc = CLIPProcessor.from_pretrained(args.model)
    templates = ["a photo of a {}", "a photo of the {}", "a {} in a house",
                 "an indoor photo of a {}", "the {} of a home"]  # 多模板平均（CLIP 慣例）
    with torch.no_grad():
        embs = []
        for t in templates:
            inp = proc(text=[t.format(c) for c in ROOM_TYPES], return_tensors="pt", padding=True)
            e = model.get_text_features(**inp)
            embs.append(e / e.norm(dim=-1, keepdim=True))
        emb = torch.stack(embs).mean(dim=0)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    np.save(args.out, emb.numpy().astype(np.float32))
    print(f"text_emb: {emb.shape} → {args.out}")
    print("classes:", ROOM_TYPES)


if __name__ == "__main__":
    main()
