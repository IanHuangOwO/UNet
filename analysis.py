import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import os

from utils.metrics import confusion_from_arrays, accumulate_confusion, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analysis")

VALID_EXTS = (".tif", ".tiff")

# Column headers
SUMMARY_HEADERS = [
    "model", "files_evaluated", "pixels_total", "tp", "fp", "fn", "tn",
    "accuracy", "precision", "recall", "f1",
    "accuracy_mean", "accuracy_std",
    "precision_mean", "precision_std",
    "recall_mean", "recall_std",
    "f1_mean", "f1_std",
]

SUBFOLDER_HEADERS = [
    "model", "subfolder", "files_evaluated", "pixels_total",
    "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1",
]

FILE_HEADERS = [
    "subfolder", "file", "pixels_total", "tp", "fp", "fn", "tn",
    "accuracy", "precision", "recall", "f1",
]

def list_subdirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir())

def list_files(path: Path, exts: Tuple[str, ...] = VALID_EXTS) -> List[Path]:
    if not path.exists():
        return []
    allowed = tuple(e.lower() for e in exts)
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in allowed)

def find_corresponding(pred_dir: Path, gt_file: Path) -> Path | None:
    cand = pred_dir / gt_file.name
    if cand.exists():
        return cand
    alt_suffix = ".tiff" if gt_file.suffix.lower() == ".tif" else ".tif"
    cand2 = pred_dir / (gt_file.stem + alt_suffix)
    if cand2.exists():
        return cand2
    return None

def to_binary(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)

def write_xlsx(
    summary_rows: List[Dict[str, object]],
    out_path: Path,
    subfolder_rows: List[Dict[str, object]] | None = None,
    per_model_file_rows: Dict[str, List[Dict[str, object]]] | None = None,
) -> None:
    try:
        import openpyxl
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        logger.warning(f"openpyxl not installed; writing CSV instead: {csv_path}")
        # Simplification: only summary for CSV fallback
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_HEADERS)
            w.writeheader()
            for r in summary_rows:
                # filter r to headers
                row = {k: r.get(k) for k in SUMMARY_HEADERS}
                w.writerow(row)
        return

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["Overall summary (per model)"])
    ws.append(SUMMARY_HEADERS)
    for r in summary_rows:
        ws.append([r.get(k, 0) for k in SUMMARY_HEADERS])

    ws.append([""])
    
    if subfolder_rows:
        ws.append(["Subfolder summary (per model, per subfolder)"])
        ws.append(SUBFOLDER_HEADERS)
        for r in subfolder_rows:
            ws.append([r.get(k, 0) for k in SUBFOLDER_HEADERS])

    if per_model_file_rows:
        for model, rows in per_model_file_rows.items():
            sheet_name = sanitize_sheet_name(model)
            ws_m = wb.create_sheet(title=sheet_name)
            ws_m.append(FILE_HEADERS)
            for r in rows:
                ws_m.append([r.get(k, 0) for k in FILE_HEADERS])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))

def sanitize_sheet_name(name: str) -> str:
    invalid = set(':\/?*[]')
    clean = ''.join(ch if ch not in invalid else '-' for ch in name)
    return clean[:31] if len(clean) > 31 else clean

def evaluate_model(base_dir: Path, gt_dir: Path, pred_dir: Path) -> Tuple[
    Dict[str, float],
    int,
    List[Dict[str, object]],
    Dict[str, Dict[str, int]],
]:
    tp = fp = fn = tn = 0
    file_count = 0
    per_file_rows: List[Dict[str, object]] = []
    per_sub_counts: Dict[str, Dict[str, int]] = {}

    for sub in list_subdirs(gt_dir):
        gt_sub = sub
        pr_sub = pred_dir / sub.name
        if not pr_sub.exists():
            continue

        gt_files = list_files(gt_sub)
        if not gt_files:
            continue

        if sub.name not in per_sub_counts:
            per_sub_counts[sub.name] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "files": 0}

        for gtf in gt_files:
            prf = find_corresponding(pr_sub, gtf)
            if prf is None:
                continue
            try:
                g_arr = to_binary(tifffile.imread(str(gtf)))
                p_arr = to_binary(tifffile.imread(str(prf)))
            except Exception as e:
                logger.warning(f"Failed to read pair {gtf} / {prf}: {e}")
                continue

            if g_arr.shape != p_arr.shape:
                continue

            tpf, fpf, fnf, tnf = confusion_from_arrays(g_arr, p_arr)
            met = compute_metrics(tpf, fpf, fnf, tnf)
            per_file_rows.append({
                "subfolder": sub.name,
                "file": gtf.name,
                **met,
            })
            c = per_sub_counts[sub.name]
            c["tp"] += tpf
            c["fp"] += fpf
            c["fn"] += fnf
            c["tn"] += tnf
            c["files"] += 1

            tp, fp, fn, tn = accumulate_confusion(tp, fp, fn, tn, g_arr, p_arr)
            file_count += 1

    metrics = compute_metrics(tp, fp, fn, tn)
    return metrics, file_count, per_file_rows, per_sub_counts

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate predicted masks against GT and export XLSX")
    p.add_argument("--base_dir", type=str, required=True, help="Base folder containing masks/ and masks_* folders")
    p.add_argument("--gt_name", type=str, default="masks", help="Ground-truth masks folder name (default: masks)")
    p.add_argument("--pred_prefix", type=str, default="masks_", help="Prefix for prediction folders (default: masks_)")
    p.add_argument("--output", type=str, default="metrics.xlsx", help="Output XLSX filename (relative to base_dir)")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    base = Path(args.base_dir)
    if not base.exists():
        logger.error(f"Base dir not found: {base}")
        return 1

    gt_dir = base / args.gt_name
    if not gt_dir.exists():
        logger.error(f"Ground-truth folder not found: {gt_dir}")
        return 1

    pred_dirs = [p for p in list_subdirs(base) if p.name.startswith(args.pred_prefix)]
    if not pred_dirs:
        logger.error(f"No prediction folders found.")
        return 1

    rows: List[Dict[str, object]] = []
    subfolder_rows: List[Dict[str, object]] = []
    per_model_file_rows: Dict[str, List[Dict[str, object]]] = {}

    for pred in pred_dirs:
        model_name = pred.name[len(args.pred_prefix):] or pred.name
        logger.info(f"Evaluating model: {model_name}")
        metrics, nfiles, per_file, per_sub = evaluate_model(base, gt_dir, pred)

        vals_acc = np.array([m.get("accuracy", 0.0) for m in per_file], dtype=float)
        vals_prec = np.array([m.get("precision", 0.0) for m in per_file], dtype=float)
        vals_rec = np.array([m.get("recall", 0.0) for m in per_file], dtype=float)
        vals_f1 = np.array([m.get("f1", 0.0) for m in per_file], dtype=float)

        summary_row = {
            "model": model_name,
            "files_evaluated": nfiles,
            **metrics,
            "accuracy_mean": float(vals_acc.mean()) if vals_acc.size else 0.0,
            "accuracy_std": float(vals_acc.std()) if vals_acc.size else 0.0,
            "precision_mean": float(vals_prec.mean()) if vals_prec.size else 0.0,
            "precision_std": float(vals_prec.std()) if vals_prec.size else 0.0,
            "recall_mean": float(vals_rec.mean()) if vals_rec.size else 0.0,
            "recall_std": float(vals_rec.std()) if vals_rec.size else 0.0,
            "f1_mean": float(vals_f1.mean()) if vals_f1.size else 0.0,
            "f1_std": float(vals_f1.std()) if vals_f1.size else 0.0,
        }
        rows.append(summary_row)

        for sub_name, counts in per_sub.items():
            sub_metrics = compute_metrics(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
            subfolder_rows.append({
                "model": model_name,
                "subfolder": sub_name,
                "files_evaluated": counts.get("files", 0),
                **sub_metrics,
            })

        per_model_file_rows[model_name] = per_file

    out_path = base / args.output
    write_xlsx(rows, out_path, subfolder_rows=subfolder_rows, per_model_file_rows=per_model_file_rows)
    logger.info(f"Wrote metrics: {out_path}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
