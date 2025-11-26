from pathlib import Path
from typing import List ,Union
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# D-FINE repo path
REPO_ROOT = "/home/abuzayed.a/D-FINE"
sys.path.append(REPO_ROOT)

from src.core import YAMLConfig  # official repo loader


#  Inference wrapper
class DFineDeployModel(nn.Module):
    """Thin wrapper for repo-native model + postprocessor."""
    def __init__(self, cfg: YAMLConfig):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        # Returns (labels_list, boxes_list, scores_list) for each image in batch
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


# inference function
def run_dfine(
    model: DFineDeployModel,
    image_paths: List[Union[str, Path]],
    device: str = "cuda:0",
    batch_size: int = 16,
):
    device = torch.device(device if device != "cpu" else "cpu")
    model.to(device).eval()

    tfm = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),  # D-FINE expects [0,1] range
    ])

    all_results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        pil_list = [Image.open(str(p)).convert("RGB") for p in batch_paths]
        sizes = torch.tensor([[im.size[0], im.size[1]] for im in pil_list], dtype=torch.float32, device=device)
        imgs = torch.stack([tfm(im) for im in pil_list]).to(device)

        # Forward pass
        labels_list, boxes_list, scores_list = model(imgs, sizes)

        # Convert to dictionaries per image
        for path, labels, boxes, scores in zip(batch_paths, labels_list, boxes_list, scores_list):
            labels_np = labels.detach().cpu().numpy().astype(int)
            boxes_np = boxes.detach().cpu().numpy().astype(float)
            scores_np = scores.detach().cpu().numpy().astype(float)
            all_results.append({
                "image": str(path),
                "labels": labels_np.tolist(),
                "boxes": boxes_np.tolist(),
                "scores": scores_np.tolist(),
            })

    return all_results


#  Load model from YAML + checkpoint 
def load_dfine_model(config_path: str, ckpt_path: str, device: str = "cuda:0") -> DFineDeployModel:
    cfg = YAMLConfig(config_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cfg.model.load_state_dict(state, strict=False)

    model = DFineDeployModel(cfg).to(device).eval()
    return model



config_path = "/home/abuzayed.a/D-FINE/configs/dfine/dfine_hgnetv2_l_coco_layout.yml"
ckpt_path = "/home/shared_storage/TalkToDocs-Ingest/models/layout/dfine_batch_0_1_2_3_4_5_6_7_8_doclaynet/best_stg1.pth"
device= "cuda:6"

image_path= ["/home/abuzayed.a/Notebooks/chart_example_1.png", "/home/abuzayed.a/Notebooks/chart_example_1.png"]

# Load model
model = load_dfine_model(config_path,ckpt_path=ckpt_path, device=device)

# Run inference
results = run_dfine(model,image_path, device= device, batch_size=16)

for r in results:
        print(r)
