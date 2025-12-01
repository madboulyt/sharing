### this code has been adapted from Hossam's eval.py with some modifications
"""
Two-stage evaluation for D-FINE:
  1) Run on VAL set to find best global confidence threshold
  2) Apply that threshold to TEST and produce final metrics 

D-FINE (repo-native) evaluation with automatic confidence-threshold selection.

What you get (same as before):
- F1/Fβ-optimal global threshold (and optional per-class thresholds)
- Saved curves (P, R, F1 vs conf; PR curve; Fβ vs conf)
- Confusion matrix at your chosen deployment conf
- best_thresholds.json to reuse in inference

This version uses the OFFICIAL D-FINE repo runtime:
- YAMLConfig to build model & postprocessor
- model.deploy() and postprocessor.deploy()
- Forward pass: outputs = model(images); (labels, boxes, scores) = postprocessor(outputs, orig_sizes)

Refs: Official D-FINE repo, configs, and usage.  (See README / src/core / tools) 
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Optional, Union

import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import supervision as sv
from supervision.metrics import MeanAveragePrecision, F1Score, MeanAverageRecall, Precision, Recall

import torch
import torch.nn as nn
import torchvision.transforms as T

# --- D-FINE repo imports (assumes you run inside the D-FINE repo or add it to PYTHONPATH)
# If this file is NOT inside the repo, uncomment the next 2 lines and point to your clone:
REPO_ROOT = "/home/abuzayed.a/D-FINE"
sys.path.append(REPO_ROOT)
from src.core import YAMLConfig  # provided by the official repo


# ───────────────────────────────────────────────────────────────
#  Dataset helpers
# ───────────────────────────────────────────────────────────────
def convert_coco_to_sv_format(
    coco_labels: List[Dict],
    classes_: Sequence[str]
) -> sv.Detections:
    if not coco_labels:
        return sv.Detections.empty()

    xyxy = np.asarray(
        [[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
         for bb in (ann["bbox"] for ann in coco_labels)],
        dtype=np.float32,
    )
    # Assumes your COCO has category_id already 0..K-1 (as you said)
    class_ids = np.asarray([ann["category_id"] for ann in coco_labels], dtype=np.int32)

    return sv.Detections(
        xyxy=xyxy,
        class_id=class_ids,
        data={"class_name": np.array([classes_[cid] for cid in class_ids], dtype=object)},
        confidence=np.ones(len(class_ids), dtype=np.float32),
    )


def read_coco_categories(coco_json: str) -> List[str]:
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in cats]


def prepare_gt_data_coco_for_sv(
    coco_json: str,
    img_dir: str,
    classes_: Sequence[str]
) -> Tuple[List[Path], List[sv.Detections]]:
    with open(coco_json, "r", encoding="utf-8") as f:
        data_coco = json.load(f)

    image_id_to_anns: Dict[int, List[Dict]] = defaultdict(list)
    for ann in data_coco["annotations"]:
        image_id_to_anns[ann["image_id"]].append(ann)

    paths: List[Path] = []
    dets: List[sv.Detections] = []
    for img_meta in tqdm(data_coco["images"], desc="Load GT"):
        p = Path(img_dir) / img_meta["file_name"]
        paths.append(p)
        dets.append(convert_coco_to_sv_format(image_id_to_anns[img_meta["id"]], classes_))
    return paths, dets


# ───────────────────────────────────────────────────────────────
#  OpenCV drawer (optional eyeballing)
# ───────────────────────────────────────────────────────────────
def draw_boxes(
    classes_: Sequence[str],
    img: np.ndarray,
    det: sv.Detections,
    color: Tuple[int, int, int],
    show_conf: bool = False
) -> np.ndarray:
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(det.xyxy.astype(int)):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        if show_conf:
            cls_name = classes_[int(det.class_id[i])] if det.class_id is not None else "?"
            conf = float(det.confidence[i]) if det.confidence is not None else 1.0
            text = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1 - 5, th + 2)
            cv2.rectangle(out, (x1, y_text - th - 2), (x1 + tw, y_text), color, -1)
            cv2.putText(out, text, (x1, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
    return out


# ───────────────────────────────────────────────────────────────
#  IoU, matching, curves, confusion matrix
# ───────────────────────────────────────────────────────────────
def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def _greedy_match(gt: sv.Detections, pr: sv.Detections, iou_thr: float = 0.50):
    if len(pr) == 0:
        return np.zeros(0, dtype=np.uint8), np.full(0, -1, dtype=np.int32)

    order = np.argsort(-pr.confidence)
    pr_sorted = pr[order]

    ious = iou_matrix(gt.xyxy, pr_sorted.xyxy) if len(gt) and len(pr_sorted) else np.zeros((len(gt), len(pr_sorted)))
    gt_used = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pr_sorted), dtype=np.uint8)
    assign_gt = np.full(len(pr_sorted), -1, dtype=np.int32)

    for j in range(len(pr_sorted)):
        gi = -1
        best = iou_thr
        for g in range(len(gt)):
            if gt_used[g]:
                continue
            if gt.class_id is not None and pr_sorted.class_id is not None:
                if int(gt.class_id[g]) != int(pr_sorted.class_id[j]):
                    continue
            if ious[g, j] >= best:
                best = ious[g, j]
                gi = g
        if gi >= 0:
            gt_used[gi] = True
            tp[j] = 1
            assign_gt[j] = gi

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    tp_final = tp[inv]
    assign_final = assign_gt[inv]
    return tp_final, assign_final


def _counts_at_threshold(all_gt, all_pr, conf_thr: float, iou_thr: float = 0.50):
    TP = FP = FN = 0
    for gt, pr in zip(all_gt, all_pr):
        if len(pr):
            keep = pr.confidence >= conf_thr
            pr_t = pr[keep] if np.any(keep) else sv.Detections.empty()
        else:
            pr_t = sv.Detections.empty()
        tp_vec, _ = _greedy_match(gt, pr_t, iou_thr=iou_thr)
        tp = int(tp_vec.sum())
        fp = int(len(pr_t) - tp)
        fn = int(len(gt) - tp)
        TP += tp; FP += fp; FN += fn
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_curves(all_gt, all_pr, save_dir: Path, iou_thr: float = 0.50):
    save_dir.mkdir(parents=True, exist_ok=True)
    conf_grid = np.linspace(0.0, 1.0, 101)

    P, R, F1 = [], [], []
    for c in conf_grid:
        p, r, f = _counts_at_threshold(all_gt, all_pr, c, iou_thr=iou_thr)
        P.append(p); R.append(r); F1.append(f)
    P = np.array(P); R = np.array(R); F1 = np.array(F1)

    plt.figure(); plt.plot(conf_grid, P); plt.xlabel("Confidence threshold"); plt.ylabel("Precision")
    plt.title(f"P_curve (IoU={iou_thr:.2f})"); plt.grid(True)
    plt.savefig(save_dir / "P_curve.png", dpi=200); plt.close()

    plt.figure(); plt.plot(conf_grid, R); plt.xlabel("Confidence threshold"); plt.ylabel("Recall")
    plt.title(f"R_curve (IoU={iou_thr:.2f})"); plt.grid(True)
    plt.savefig(save_dir / "R_curve.png", dpi=200); plt.close()

    plt.figure(); plt.plot(conf_grid, F1); plt.xlabel("Confidence threshold"); plt.ylabel("F1")
    plt.title(f"F1_curve (IoU={iou_thr:.2f})"); plt.grid(True)
    plt.savefig(save_dir / "F1_curve.png", dpi=200); plt.close()

    order = np.argsort(R)
    plt.figure(); plt.plot(R[order], P[order]); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR_curve (IoU={iou_thr:.2f})"); plt.grid(True)
    plt.savefig(save_dir / "PR_curve.png", dpi=200); plt.close()

# TODO: check and use supervision confusion matrix function 
def confusion_matrix_at_threshold(
    all_gt,
    all_pr,
    conf_thr: float,
    iou_thr: float = 0.50,
    names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    if names is not None:
        num_classes = len(names)
    else:
        m = 0
        for d in list(all_gt) + list(all_pr):
            if d.class_id is not None and len(d.class_id):
                m = max(m, int(np.max(d.class_id)) + 1)
        num_classes = m or 1

    mat = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for gt, pr in zip(all_gt, all_pr):
        if len(pr):
            keep = pr.confidence >= conf_thr
            pr_t = pr[keep] if np.any(keep) else sv.Detections.empty()
        else:
            pr_t = sv.Detections.empty()

        tp_vec, assign_gt = _greedy_match(gt, pr_t, iou_thr=iou_thr)

        for j in np.where(tp_vec == 1)[0]:
            gi = assign_gt[j]
            if gi >= 0:
                true_c = int(gt.class_id[gi]) if gt.class_id is not None else 0
                pred_c = int(pr_t.class_id[j]) if pr_t.class_id is not None else 0
                mat[true_c, pred_c] += 1

        for j in np.where(tp_vec == 0)[0]:
            pred_c = int(pr_t.class_id[j]) if pr_t.class_id is not None else 0
            mat[num_classes, pred_c] += 1

        matched_gt_idx = set(assign_gt[tp_vec == 1].tolist())
        for gi in range(len(gt)):
            if gi not in matched_gt_idx:
                true_c = int(gt.class_id[gi]) if gt.class_id is not None else 0
                mat[true_c, num_classes] += 1

    labels = list(names) if names is not None else [str(i) for i in range(num_classes)]
    labels = labels + ["background"]

    row_sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat_pct = np.where(row_sums > 0, (mat / row_sums) * 100.0, 0.0)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix (IoU={iou_thr:.2f}, conf={conf_thr:.2f}) — counts with row %")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Count")

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    vmax = mat.max() if mat.size else 0
    thresh = vmax / 2.0 if vmax > 0 else 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            count = mat[i, j]
            pct = mat_pct[i, j]
            text = f"{int(count)} ({pct:.1f}%)"
            plt.text(
                j, i, text,
                ha="center", va="center",
                color="white" if count > thresh else "black",
                fontsize=4
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# ───────────────────────────────────────────────────────────────
#  Threshold selection
# ───────────────────────────────────────────────────────────────
def best_threshold_by_objective(
    all_gt, all_pr, iou_thr: float = 0.50, beta: float = 1.0, num_points: int = 1001
):
    conf_grid = np.linspace(0.0, 1.0, num_points)
    P, R, F = [], [], []
    b2 = beta * beta
    for c in conf_grid:
        p, r, _ = _counts_at_threshold(all_gt, all_pr, c, iou_thr=iou_thr)
        f = (1 + b2) * p * r / (b2 * p + r) if (p + r) > 0 else 0.0
        P.append(p); R.append(r); F.append(f)
    P, R, F = np.array(P), np.array(R), np.array(F)
    best_idx = int(np.argmax(F))
    return float(conf_grid[best_idx]), {"conf": conf_grid, "precision": P, "recall": R, "f": F}


def best_thresholds_per_class(
    all_gt, all_pr, names, iou_thr: float = 0.50, beta: float = 1.0, num_points: int = 301
):
    results = {}
    for cls_id, cls_name in enumerate(names):
        cls_gt, cls_pr = [], []
        for gt, pr in zip(all_gt, all_pr):
            gmask = (gt.class_id == cls_id) if gt.class_id is not None and len(gt) else np.zeros(len(gt), bool)
            pmask = (pr.class_id == cls_id) if pr.class_id is not None and len(pr) else np.zeros(len(pr), bool)
            gt_c = gt[gmask] if np.any(gmask) else sv.Detections.empty()
            pr_c = pr[pmask] if np.any(pmask) else sv.Detections.empty()
            cls_gt.append(gt_c); cls_pr.append(pr_c)
        best_c, _ = best_threshold_by_objective(cls_gt, cls_pr, iou_thr=iou_thr, beta=beta, num_points=num_points)
        results[cls_name] = best_c
    return results


# ───────────────────────────────────────────────────────────────
#  D-FINE inference core (repo-native)
# ───────────────────────────────────────────────────────────────
class DFineDeployModel(nn.Module):
    """Thin wrapper that mimics your reference code: model.deploy() + postprocessor.deploy()."""
    def __init__(self, cfg: YAMLConfig):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        """
        images: (B, 3, H, W) float tensor in [0,1], normalized by D-FINE's training pipeline if needed
        orig_target_sizes: (B, 2) tensor [(w_i, h_i), ...]
        returns: for each batch element, (labels_i, boxes_i, scores_i)
        """
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


# ───────────────────────────────────────────────────────────────
#  Evaluation + visualisation (using DFineDeployModel)
# ───────────────────────────────────────────────────────────────
def evaluate_at_thr(
    model: DFineDeployModel,
    img_paths: List[Path],
    gt_dets: List[sv.Detections],
    batch_size: int = 16,
    conf_thr_for_eyeball: float = 0.30,
    out_dir: Optional[Union[str, Path]] = None
):
    out_path: Optional[Path] = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        img_out_dir = out_path / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        img_out_dir = None

    # If you want class names in plots, pass --use_coco_categories 1 to use GT names
    # Otherwise, set your names manually here.
    # We'll reconstruct names from GT usage later in main().

    f1_metric, mAP_metric, mAR_metric, recall_metric, precision_metric = F1Score(), MeanAveragePrecision(), MeanAverageRecall(), Recall(), Precision()
    all_gt: List[sv.Detections] = []
    all_pr_unpruned: List[sv.Detections] = []

    device = next(model.parameters()).device
    tfm = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        # NOTE: Add Normalize if your training used mean/std. D-FINE configs typically
        # have their own preprocessing; ToTensor() keeps [0,1]. If your results look off,
        # plug in the exact normalization used during training.
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def batches(a, b, bs):
        for i in range(0, len(a), bs):
            yield a[i:i + bs], b[i:i + bs]

    total_batches = (len(img_paths) + batch_size - 1) // batch_size

    for img_batch, gt_batch in tqdm(batches(img_paths, gt_dets, batch_size),
                                    total=total_batches, desc="D-FINE Inference"):
        # Load & transform
        pil_list = [Image.open(str(p)).convert("RGB") for p in img_batch]
        sizes = torch.tensor([[im.size[0], im.size[1]] for im in pil_list], dtype=torch.float32, device=device)  # (w,h)
        imgs = torch.stack([tfm(im) for im in pil_list]).to(device)  # (B,3,640,640)

        # Forward (repo-native)
        labels_list, boxes_list, scores_list = model(imgs, sizes)

        # For each image, map to sv.Detections
        for path, gt, labels, boxes, scores in zip(img_batch, gt_batch, labels_list, boxes_list, scores_list):
            # Convert tensors to numpy
            labels_np = labels.detach().cpu().numpy().astype(np.int32)
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)  # expected xyxy from repo postprocessor
            scores_np = scores.detach().cpu().numpy().astype(np.float32)

            # Build detections (we don't need names here; just class_id/scores)
            pred = sv.Detections(xyxy=boxes_np, confidence=scores_np, class_id=labels_np)

            # Keep unpruned for curve-building (NMS is handled in postprocessor already; don't drop by conf here)
            all_gt.append(gt)
            all_pr_unpruned.append(pred)

            # Eyeball/summary metrics at selected conf
            pred_for_metrics = pred
            if conf_thr_for_eyeball is not None:
                pred_for_metrics = pred_for_metrics[pred_for_metrics.confidence >= conf_thr_for_eyeball]

            f1_metric.update(predictions= pred_for_metrics, targets= gt)
            mAP_metric.update(predictions= pred_for_metrics, targets= gt)
            mAR_metric.update(predictions= pred_for_metrics, targets= gt)
            precision_metric.update(predictions= pred_for_metrics, targets= gt)
            recall_metric.update(predictions= pred_for_metrics, targets= gt)


            # Optional visualizations
            #if out_path:
            #    bgr = cv2.imread(str(path))
                # Try to inject class names if saved in gt.data
            #    class_names = None
            #    if gt.data and "class_name" in gt.data:
                    # gt.data["class_name"] mirrors gt.class_id; but we only need a mapping list.
                    # We'll compose it in main() and pass around; for quick viz keep label ids.
            #        pass
            #    gt_vis = draw_boxes([str(i) for i in range(999)], bgr, gt, (0, 255, 0), show_conf=True)
            #    pd_vis = draw_boxes([str(i) for i in range(999)], bgr, pred_for_metrics, (0, 0, 255), show_conf=True)
            #    side_by_side = np.hstack([gt_vis, pd_vis])
            #    cv2.imwrite(str(img_out_dir / f"{Path(path).stem}_gt_pred.jpg"), side_by_side)

    return f1_metric.compute(), mAP_metric.compute(), mAR_metric.compute(), precision_metric.compute(),recall_metric.compute() , all_gt, all_pr_unpruned

def visualize_predictions(
    img_paths: list,
    gt_list: list,
    pred_list: list,
    classes_: list,
    conf_thr: float,
    save_dir: Union[str, Path],
    desc: str = "Visualization"
):
    """
    Save side-by-side ground-truth and predictions for a dataset.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for path, gt, pr in tqdm(
        zip(img_paths, gt_list, pred_list),
        total=len(img_paths),
        desc=desc
    ):
        # Apply confidence threshold
        if len(pr):
            keep = pr.confidence >= conf_thr
            pr_t = pr[keep] if np.any(keep) else sv.Detections.empty()
        else:
            pr_t = sv.Detections.empty()

        # Load image
        bgr = cv2.imread(str(path))

        # Draw boxes
        gt_vis = draw_boxes(classes_, bgr, gt, (0, 255, 0), show_conf=True)
        pd_vis = draw_boxes(classes_, bgr, pr_t, (0, 0, 255), show_conf=True)

        # Side-by-side visualization
        side_by_side = np.hstack([gt_vis, pd_vis])

        cv2.imwrite(str(save_dir /  f"{Path(path).stem}_gt_pred.jpg"), side_by_side)


    print(f"[saved] Visualization images to {save_dir}")


# ───────────────────────────────────────────────────────────────
#  Execution
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # COCO GT
   # parser.add_argument("--gt_coco_json", type=str, required=True)
   # parser.add_argument("--gt_coco_imgs_dir", type=str, required=True)
    parser.add_argument("--val_coco_json", type=str, required=True)
    parser.add_argument("--val_coco_imgs_dir", type=str, required=True)
    parser.add_argument("--test_coco_json", type=str, required=True)
    parser.add_argument("--test_coco_imgs_dir", type=str, required=True)
    # D-FINE repo bits
    parser.add_argument("-c", "--config", type=str, required=True, help="D-FINE YAML config path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint (.pth) trained with D-FINE repo")

    # Eval controls
    parser.add_argument("--use_coco_categories", type=int, default=1,
                        help="If 1, class names are pulled from GT JSON's categories")
    parser.add_argument("--eval_output", type=str, default="./eval_output")
    parser.add_argument("--conf_thr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iou_for_curves", type=float, default=0.50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--per_class", type=int, default=1)

    args = parser.parse_args()

    # ---------------------- Load class names ----------------------
    if args.use_coco_categories:
        classes_ = read_coco_categories(args.val_coco_json)
        print(f"[info] Using categories from COCO GT ({len(classes_)} classes)")
    else:
        # Fallback: numeric names
        # (If you prefer, you can read num_classes from YAMLConfig and build a numeric list)
        classes_ = None

    # ---------------------- Build GT ----------------------
    if classes_ is None:
        # When class names unknown, still load GT; we can't create names but metrics use ids
        with open(args.val_coco_json, "r", encoding="utf-8") as f:
            data_coco = json.load(f)
        cats = sorted(data_coco.get("categories", []), key=lambda c: c["id"])
        classes_ = [str(c["id"]) for c in cats]  # simple numeric strings
        print(f"[info] Using numeric class names from GT ids ({len(classes_)} classes)")

    #paths, gt = prepare_gt_data_coco_for_sv(
    #    args.gt_coco_json,
    #    args.gt_coco_imgs_dir,
    #    classes_=classes_)

    # Prepare GT for val & test
    val_paths, val_gt = prepare_gt_data_coco_for_sv(args.val_coco_json, args.val_coco_imgs_dir, classes_=classes_)
    test_paths, test_gt = prepare_gt_data_coco_for_sv(args.test_coco_json, args.test_coco_imgs_dir, classes_=classes_)

    # ---------------------- D-FINE repo: load & deploy ----------------------
    cfg = YAMLConfig(args.config, resume=args.resume)
    # (Optional) avoid loading external pretrained backbone when resuming
    if "HGNetv2" in cfg.yaml_cfg:
        # the repo often puts pretrained=True; when resuming, disable backbone external preload
        try:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        except Exception:
            pass

    # Load weights from checkpoint (ema/model keys are used in repo)
    checkpoint = torch.load(args.resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        # trust raw state
        state = checkpoint
    cfg.model.load_state_dict(state, strict=True)

    # Build deploy model + postprocessor
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    d_model = DFineDeployModel(cfg).to(device)
    d_model.eval()

    # ---------------------- Evaluate ----------------------

    # 1) Run on VAL to get best threshold
    f1, mAP, mAR, p, r, val_all_gt, val_all_pred = evaluate_at_thr(
        d_model,
        val_paths,
        val_gt,
        batch_size=args.batch_size,
        conf_thr_for_eyeball=args.conf_thr,
        out_dir= None
    )

    # ---------------------- Report + curves ----------------------
    print(f"\nSupervision (with conf_thr={args.conf_thr:.3f} for metrics on VAL set):")
    print(mAP)
    print(f"mAP.map50      : {mAP.map50:.4f}")
    print(f"mAP.map75      : {mAP.map75:.4f}")
    print(f"mAP.map50_95   : {mAP.map50_95:.4f}")
   # print(mAR)
    print(f"mAR.mAR_at_1   : {mAR.mAR_at_1:.4f}")
    print(f"mAR.mAR_at_10  : {mAR.mAR_at_10:.4f}")
    print(f"mAR.mAR_at_100 : {mAR.mAR_at_100:.4f}")

    print(f"f1.f1_50        : {f1.f1_50:.4f}")
    print(f"f1.f1_75        : {f1.f1_75:.4f}")

    #print(f"Precision       : {p}")
    print(f"Precision_50    : {p.precision_at_50:.4f}")
    print(f"Precision_75    : {p.precision_at_75:.4f}")
    
    #print(f"Recall          : {r}")
    print(f"Recall_50       : {r.recall_at_50:.4f}")
    print(f"Recall_75       : {r.recall_at_75:.4f}")

    curves_dir = Path(args.eval_output or ".") / "ultralytics_like_curves"
    build_curves(val_all_gt, val_all_pred, save_dir=curves_dir, iou_thr=args.iou_for_curves)
    print(f"[saved] {curves_dir}/P_curve.png")
    print(f"[saved] {curves_dir}/R_curve.png")
    print(f"[saved] {curves_dir}/F1_curve.png")
    print(f"[saved] {curves_dir}/PR_curve.png")

    ## Select best global threshold
    print("\n Selecting best GLOBAL confidence threshold...")
    best_conf, fcurve = best_threshold_by_objective(
        val_all_gt, val_all_pred, iou_thr=args.iou_for_curves, beta=args.beta, num_points=1001
    )
    print(f"\n[selection] Best GLOBAL confidence (F{args.beta:.1f} @ IoU={args.iou_for_curves:.2f}): {best_conf:.3f}")

    ##  Visualizations at best threshold
    if args.eval_output:
        visualize_predictions(
            img_paths=val_paths,
            gt_list=val_all_gt,
            pred_list=val_all_pred,
            classes_=classes_,
            conf_thr=best_conf,
            save_dir=Path(args.eval_output) / "val_images",
            desc="VAL Best-threshold visualization"
        )

    print("\n Plotting F-beta curve...")
    plt.figure()
    plt.plot(fcurve["conf"], fcurve["f"])
    plt.axvline(best_conf, linestyle="--")
    plt.xlabel("Confidence")
    plt.ylabel(f"F{args.beta:.1f} (IoU={args.iou_for_curves:.2f})")
    plt.title("F-beta vs Confidence")
    plt.grid(True)
    plt.savefig(curves_dir / "Fbeta_curve.png", dpi=200); plt.close()
    print(f"[saved] {curves_dir}/Fbeta_curve.png")

    print("\n Selecting best PER-CLASS confidence thresholds...")
    per_class = {}
    if args.per_class:
        per_class = best_thresholds_per_class(
            val_all_gt, val_all_pred, names=list(read_coco_categories(args.val_coco_json)),
            iou_thr=args.iou_for_curves, beta=args.beta, num_points=301
        )
        print("\n[selection] Per-class best confidence:")
        for k, v in per_class.items():
            print(f"  {k:>16}: {v:.3f}")

    cm_path = curves_dir / "confusion_matrix.png"
    confusion_matrix_at_threshold(
        val_all_gt, val_all_pred,
        conf_thr=best_conf,
        iou_thr=args.iou_for_curves,
        names=list(read_coco_categories(args.val_coco_json)),
        save_path=cm_path
    )
    print(f"[saved] {cm_path}")

## 2) Evaluate on TEST set using selected best threshold on VAL
    print(f"\nEvaluating TEST set at best GLOBAL confidence threshold {best_conf:.3f}...")
    f1_test, mAP_test, mAR_test, p_test, r_test, test_all_gt, test_all_pred = evaluate_at_thr(
        d_model,
        test_paths,
        test_gt,
        batch_size=args.batch_size,
        conf_thr_for_eyeball=best_conf,
        out_dir= None
    )

    # ---------------------- Report ----------------------
    print(f"\nSupervision (with conf_thr={best_conf:.3f} for metrics on TEST set):")
    print(mAP_test)
    print(f"mAP.map50      : {mAP_test.map50:.4f}")
    print(f"mAP.map75      : {mAP_test.map75:.4f}")
    print(f"mAP.map50_95   : {mAP_test.map50_95:.4f}")
   # print(mAR_test)
    print(f"mAR.mAR_at_1   : {mAR_test.mAR_at_1:.4f}")
    print(f"mAR.mAR_at_10  : {mAR_test.mAR_at_10:.4f}")
    print(f"mAR.mAR_at_100 : {mAR_test.mAR_at_100:.4f}")

    print(f"f1.f1_50        : {f1_test.f1_50:.4f}")
    print(f"f1.f1_75        : {f1_test.f1_75:.4f}")

    #print(f"Precision       : {p_test}")
    print(f"Precision_50    : {p_test.precision_at_50:.4f}")
    print(f"Precision_75    : {p_test.precision_at_75:.4f}")
    
    #print(f"Recall          : {r_test}")
    print(f"Recall_50       : {r_test.recall_at_50:.4f}")
    print(f"Recall_75       : {r_test.recall_at_75:.4f}")


    ##  Visualizations at best threshold
    if args.eval_output:
        visualize_predictions(
            img_paths=test_paths,
            gt_list=test_all_gt,
            pred_list=test_all_pred,
            classes_=classes_,
            conf_thr=best_conf,
            save_dir=Path(args.eval_output) / "test_images",
            desc="TEST Best-threshold visualization"
        )

    # ---------------------- Save results ----------------------
    results_dir = Path(args.eval_output)
    result_path = results_dir / "results_on_test.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "global_best_conf": best_conf,
            "per_class_best_conf": per_class,
            "iou": args.iou_for_curves,
            "beta": args.beta,
            "supervision_metrics": {
            "f1_f1_50": f1_test.f1_50,
            "f1_f1_75": f1_test.f1_75,
            "precision_50": p_test.precision_at_50,
            "precision_75": p_test.precision_at_75,
            "recall_50": r_test.recall_at_50,
            "recall_75": r_test.recall_at_75,
            "map50": mAP_test.map50,
            "map75": mAP_test.map75,
            "map50_95": mAP_test.map50_95,
            "mAR_at_1": mAR_test.mAR_at_1,
            "mAR_at_10": mAR_test.mAR_at_10,
            "mAR_at_100": mAR_test.mAR_at_100
            }
        }, f, indent=2)
    print(f"[saved] {result_path}")


"""
   python eval_dfine.py \
   --config "/home/abuzayed.a/D-FINE/configs/dfine/dfine_hgnetv2_l_coco_layout.yml" \
   --resume "/home/shared_storage/TalkToDocs-Ingest/models/layout/dfine_batch_0_1_2_3_4_5_6_7_8_doclaynet/best_stg1.pth"  \
   --val_coco_imgs_dir /home/shared_storage/TalkToDocs-Ingest/data/layout/merged_data/dataset5_batch0_2_3_4_5_6_7_8/val/images \
   --val_coco_json /home/shared_storage/TalkToDocs-Ingest/data/layout/merged_data/dataset5_batch0_2_3_4_5_6_7_8/val/annotations.coco.json \
   --test_coco_imgs_dir /home/shared_storage/TalkToDocs-Ingest/data/layout/merged_data/dataset5_batch0_2_3_4_5_6_7_8/test/images \
   --test_coco_json /home/shared_storage/TalkToDocs-Ingest/data/layout/merged_data/dataset5_batch0_2_3_4_5_6_7_8/test/annotations.coco.json \
   --eval_output /home/shared_storage/TalkToDocs-Ingest/data/layout/eval_results/dfine_batch_0_1_2_3_4_5_6_7_8_doclaynet_results\
   --conf_thr  0.001\
   --device cuda:6 \
   --batch_size 64 \
   --iou_for_curves 0.50 \
   --beta 1.0 \
   --per_class 1
"""
