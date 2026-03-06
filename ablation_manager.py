#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Experiment Manager for Lens Defect Detection
=====================================================
Manages 6 ablation experiments by toggling three core modules:
  - Semi-supervised anomaly synthesis  (real_feat_guidance + use_real_in_image_synth)
  - Cost filter / MR-CFN              (use_costfilter)
  - Polar coordinate transform         (use_polar)

Usage:
    python ablation_manager.py \
        --datapath /path/to/lens/OK_ROI0227 \
        --augpath  /path/to/dtd/images \
        --realbank /path/to/lens_semi_anomaly \
        --realimg  /path/to/lens_semi_anomaly \
        --classes utd_data \
        --gpu 0 \
        --output_root outputs

    # Run only specific experiments:
    python ablation_manager.py ... --exp_ids full A1

    # Dry-run (print commands without executing):
    python ablation_manager.py ... --dry_run
"""

import argparse
import os
import subprocess
import sys
from collections import OrderedDict

import pandas as pd

# ════════════════════════════════════════════════════════════
# Experiment Definitions
# ════════════════════════════════════════════════════════════
#   semi  = real_feat_guidance + use_real_in_image_synth
#   cf    = use_costfilter
#   polar = use_polar  (polar_ring_constraint follows polar)

EXPERIMENTS = OrderedDict([
    ("full", {
        "desc": "Full (Ours): all three modules enabled",
        "semi": True,
        "cf": True,
        "polar": True,
    }),
    ("A1_no_semi", {
        "desc": "A1: disable semi-supervised synthesis → verify near-distribution recognition",
        "semi": False,
        "cf": True,
        "polar": True,
    }),
    ("A2_no_costfilter", {
        "desc": "A2: disable cost filter (MR-CFN) → verify spatial refinement contribution",
        "semi": True,
        "cf": False,
        "polar": True,
    }),
    ("A3_no_polar", {
        "desc": "A3: disable polar transform → verify circular topology alignment",
        "semi": True,
        "cf": True,
        "polar": False,
    }),
    ("A4_only_polar", {
        "desc": "A4: only polar transform → baseline with geometric prior only",
        "semi": False,
        "cf": False,
        "polar": True,
    }),
    ("A5_baseline", {
        "desc": "A5 (Baseline): all three modules disabled → performance lower-bound",
        "semi": False,
        "cf": False,
        "polar": False,
    }),
])


def build_command(args, exp_id, exp_cfg):
    """Build the full `python main.py net dataset` command list for one experiment."""

    results_path = os.path.join(args.output_root, exp_id)

    # --- Boolean switches ---
    semi = exp_cfg["semi"]
    cf = exp_cfg["cf"]
    polar = exp_cfg["polar"]

    use_real_feat_guidance = 1 if semi else 0
    use_real_in_image_synth = 1 if semi else 0
    use_costfilter = 1 if cf else 0
    use_polar = 1 if polar else 0
    polar_ring_constraint = 1 if polar else 0
    # synth_in_cartesian and region_split_eval are ALWAYS 1
    synth_in_cartesian = 1
    region_split_eval = 1

    # --- Subdataset flags ---
    class_flags = []
    for cls in args.classes:
        class_flags.extend(["-d", cls])

    # --- Real bank / real image paths (only meaningful when semi=True) ---
    real_bank_path = args.realbank if semi else ""
    real_anomaly_source_path = args.realimg if semi else ""
    real_anomaly_prob = "0.5" if semi else "0.0"

    cmd = [
        sys.executable, "main.py",
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
        "--test", "ckpt",
        "--results_path", results_path,
        # ── net subcommand ──
        "net",
        "-b", "wideresnet50",
        "-le", "layer2", "-le", "layer3",
        "--pretrain_embed_dimension", "1536",
        "--target_embed_dimension", "1536",
        "--patchsize", "3",
        "--meta_epochs", str(args.meta_epochs),
        "--eval_epochs", str(args.eval_epochs),
        "--dsc_layers", "2",
        "--dsc_hidden", "1024",
        "--pre_proj", "1",
        "--mining", "1",
        "--noise", "0.015",
        "--radius", "0.75",
        "--p", "0.5",
        "--step", "20",
        "--limit", "392",
        "--use_costfilter", str(use_costfilter),
        "--cf_kernel_size", "3",
        "--cf_base_channels", "32",
        "--cf_lambda", "0.2",
        "--cf_weight", "0.3",
        "--real_feat_guidance", str(use_real_feat_guidance),
        "--real_bank_path", real_bank_path,
        "--real_mode", "hybrid",
        "--real_lambda", "0.1",
        "--real_warmup_epochs", "0",
        "--real_bank_max_samples", "2048",
        "--real_mix_prob_min", "0.15",
        "--real_mix_prob_max", "0.30",
        "--real_curriculum_ratio", "0.30",
        # ── dataset subcommand ──
        "dataset",
        "--distribution", "2",
        "--mean", "0.5",
        "--std", "0.1",
        "--fg", "0",
        "--use_real_in_image_synth", str(use_real_in_image_synth),
        "--real_anomaly_source_path", real_anomaly_source_path,
        "--real_anomaly_prob", real_anomaly_prob,
        "--use_polar", str(use_polar),
        "--synth_in_cartesian", str(synth_in_cartesian),
        "--polar_inner_ratio", "0.0",
        "--polar_outer_ratio", "1.0",
        "--polar_ring_constraint", str(polar_ring_constraint),
        "--region_split_eval", str(region_split_eval),
        "--aperture_ratio", "0.25",
        "--region_center_x", "-1.0",
        "--region_center_y", "-1.0",
        "--vis_save_size", "512",
        "--rand_aug", "1",
        "--batch_size", str(args.batch_size),
        "--resize", str(args.imagesize),
        "--imagesize", str(args.imagesize),
    ]
    cmd.extend(class_flags)
    cmd.extend(["mvtec", args.datapath, args.augpath])

    return cmd


def print_experiment_table():
    """Print a summary table of all ablation experiments."""
    header = f"{'Exp ID':<20} {'Semi':^6} {'CF':^6} {'Polar':^6}  Description"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for exp_id, cfg in EXPERIMENTS.items():
        s = "ON" if cfg["semi"] else "OFF"
        c = "ON" if cfg["cf"] else "OFF"
        p = "ON" if cfg["polar"] else "OFF"
        print(f"{exp_id:<20} {s:^6} {c:^6} {p:^6}  {cfg['desc']}")
    print("=" * len(header))


def summarize_results(output_root):
    """Collect results.csv from each experiment and produce ablation_summary.csv."""
    all_frames = []

    for exp_id in EXPERIMENTS:
        csv_path = os.path.join(output_root, exp_id, "results.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {csv_path} not found.")
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "exp_id", exp_id)
        all_frames.append(df)

    if len(all_frames) == 0:
        print("\nNo results.csv files found. Nothing to summarize.")
        return

    summary = pd.concat(all_frames, ignore_index=True)
    out_path = os.path.join(output_root, "ablation_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n{'=' * 72}")
    print(f"Ablation summary saved to: {out_path}")
    print(f"{'=' * 72}")

    # Print a Markdown-style comparison table for key metrics
    key_cols = [
        "exp_id", "Row Names",
        "image_auroc", "image_ap", "image_f1_max",
        "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_pro",
    ]
    available = [c for c in key_cols if c in summary.columns]
    table = summary[available].copy()

    # Format numeric columns to percentage
    for col in available:
        if col in ("exp_id", "Row Names", "best_epoch"):
            continue
        if col in table.columns:
            table[col] = table[col].apply(
                lambda v: f"{v * 100:.2f}" if isinstance(v, (int, float)) else v
            )

    print("\n" + table.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Experiment Manager for Lens Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datapath", type=str, required=True,
                        help="Path to the lens dataset (e.g. /root/autodl-tmp/lens/OK_ROI0227)")
    parser.add_argument("--augpath", type=str, required=True,
                        help="Path to DTD augmentation images (e.g. /root/autodl-tmp/dtd/images)")
    parser.add_argument("--realbank", type=str, default="",
                        help="Path to real anomaly feature bank (for semi-supervised)")
    parser.add_argument("--realimg", type=str, default="",
                        help="Path to real anomaly images for image-level synthesis")
    parser.add_argument("--classes", type=str, nargs="+", default=["utd_data"],
                        help="Subdataset class names (default: utd_data)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--output_root", type=str, default="outputs",
                        help="Root directory for all ablation outputs (default: outputs)")
    parser.add_argument("--meta_epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--eval_epochs", type=int, default=10,
                        help="Evaluate every N epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--imagesize", type=int, default=512,
                        help="Image size for resize and crop (default: 512)")
    parser.add_argument("--exp_ids", type=str, nargs="*", default=None,
                        help="Run only these experiments (default: all). "
                             "Choices: " + ", ".join(EXPERIMENTS.keys()))
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip training, only aggregate existing results")

    args = parser.parse_args()

    # Validate selected experiment IDs
    if args.exp_ids is not None:
        for eid in args.exp_ids:
            if eid not in EXPERIMENTS:
                parser.error(f"Unknown experiment ID: '{eid}'. "
                             f"Choose from: {', '.join(EXPERIMENTS.keys())}")
        selected = args.exp_ids
    else:
        selected = list(EXPERIMENTS.keys())

    print_experiment_table()

    if args.summary_only:
        summarize_results(args.output_root)
        return

    # ──── Run experiments sequentially ────
    total = len(selected)
    for idx, exp_id in enumerate(selected, 1):
        cfg = EXPERIMENTS[exp_id]
        print(f"\n{'#' * 72}")
        print(f"# [{idx}/{total}] Experiment: {exp_id}")
        print(f"# {cfg['desc']}")
        print(f"# Semi={cfg['semi']}  CF={cfg['cf']}  Polar={cfg['polar']}")
        print(f"{'#' * 72}\n")

        cmd = build_command(args, exp_id, cfg)
        cmd_str = " ".join(cmd)

        if args.dry_run:
            print(f"[DRY RUN] {cmd_str}\n")
            continue

        # Create output directory
        exp_out = os.path.join(args.output_root, exp_id)
        os.makedirs(exp_out, exist_ok=True)

        # Save command to log file
        with open(os.path.join(exp_out, "command.txt"), "w", encoding="utf-8") as f:
            f.write(cmd_str + "\n")

        # Execute
        print(f"Running: {cmd_str}\n")
        ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if ret.returncode != 0:
            print(f"\n[ERROR] Experiment '{exp_id}' exited with code {ret.returncode}.")
            print("Continuing to next experiment...\n")
        else:
            print(f"\n[OK] Experiment '{exp_id}' completed successfully.\n")

    # ──── Summarize all results ────
    if not args.dry_run:
        summarize_results(args.output_root)


if __name__ == "__main__":
    main()
