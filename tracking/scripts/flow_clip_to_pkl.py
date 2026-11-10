"""
Generate a PHC-compatible motion PKL from one or two flow models on a real HDF5 clip.

Retargets flow joint positions to SMPL pose_aa via Humanoid_Batch gradient fit,
then saves a PKL that PHC can track directly.

Single model:
  python scripts/flow_clip_to_pkl.py \
    --hdf5    /path/to/omomo.h5 --clip 45 \
    --ckpt-a  ../generation/checkpoints/joint_flow_eef_omomo_prevframe/best.pt \
    --stats-a ../generation/checkpoints/joint_flow_eef_omomo_prevframe/stats.json \
    --label-a EEF \
    --out     data/generated/sub3lb003_eef.pkl

Two models (combined PKL — use with env.num_envs=2 for side-by-side tracking):
  python scripts/flow_clip_to_pkl.py \
    --hdf5    /path/to/omomo.h5 --clip 45 \
    --ckpt-a  ../generation/checkpoints/joint_flow_eef_omomo_prevframe/best.pt \
    --stats-a ../generation/checkpoints/joint_flow_eef_omomo_prevframe/stats.json \
    --label-a EEF \
    --ckpt-b  ../generation/checkpoints/joint_flow_obj_accad_omomo/best.pt \
    --stats-b ../generation/checkpoints/joint_flow_obj_accad_omomo/stats.json \
    --label-b OBJ \
    --out     data/generated/sub3lb003_eef_obj.pkl

Then track with PHC (single):
  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp robot=smpl_humanoid \
    robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=data/generated/sub3lb003_eef.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1

Side-by-side (two clips in one PKL):
  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp robot=smpl_humanoid \
    robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=data/generated/sub3lb003_eef_obj.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=2 headless=False test=True epoch=-1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

PHC_ROOT       = Path(__file__).resolve().parents[1]
FLOW_ROOT      = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_ROOT / "scripts"))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",    required=True)
    p.add_argument("--clip",    type=int, default=0)
    p.add_argument("--ckpt-a",  required=True)
    p.add_argument("--stats-a", required=True)
    p.add_argument("--label-a", default="model_a")
    p.add_argument("--ckpt-b",  default=None)
    p.add_argument("--stats-b", default=None)
    p.add_argument("--label-b", default="model_b")
    p.add_argument("--out",     required=True)
    p.add_argument("--humanoid-xml", default="final_humanoid.xml")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--fit-iters", type=int, default=500)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--retarget-device", default="cpu")
    return p.parse_args()


def load_clip(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name  = clips[clip_idx]
        grp   = f[f"clips/{name}"]
        ts    = grp["task_space"][:]
        jp    = grp["joint_positions"][:]
        fps   = float(grp.attrs["fps"])
        obj_rel = grp["obj_com_pos_rel"][:].astype(np.float32) if "obj_com_pos_rel" in grp else None
    print(f"Clip [{clip_idx}]: {name}  |  {len(ts)} frames @ {fps:.0f} fps  |  has_obj={obj_rel is not None}")
    return ts, jp, fps, name, obj_rel


def load_model(ckpt_path, stats, device):
    from model import JointFlowMLP
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window   = stats.get("window", 10)
    prev_dim = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim   = (3 if stats.get("no_eef", False) else 9) + (3 if stats.get("use_obj", False) else 0)
    cond_dim = prev_dim + window * ts_dim
    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, window


def run_inference(ckpt_path, stats_path, ts_world, jp_gt, obj_rel, device, ode_steps, seed):
    from animate import run_flow
    with open(stats_path) as f:
        stats = json.load(f)
    model, window = load_model(ckpt_path, stats, device)
    torch.manual_seed(seed)
    pred = run_flow(
        model, stats, ts_world, window, device,
        steps=ode_steps, fps=30.0,
        init_pelvis_xy=jp_gt[0, 0, :2],
        obj_rel=obj_rel,
    )
    return pred  # (T, 24, 3) world-frame


def retarget_to_pkl_entry(positions, fk, device, iters, fps, label):
    from flow_retarget import fit_sequence
    print(f"  Retargeting {label} ({len(positions)} frames, {iters} iters) …")
    t0  = time.perf_counter()
    fit = fit_sequence(positions, fk, device=device, iters=iters, verbose=False)
    print(f"  Done: {time.perf_counter()-t0:.1f}s  fit_err={fit['fit_err_m']*100:.2f} cm")
    return {
        "pose_quat_global":  fit["pose_quat_global"],
        "pose_quat":         fit["pose_quat"],
        "root_trans_offset": torch.from_numpy(fit["root_trans_offset"].astype(np.float64)),
        "trans_orig":        fit["root_trans_offset"].astype(np.float32),
        "beta":              np.zeros(10, dtype=np.float64),
        "gender":            "neutral",
        "pose_aa":           fit["pose_aa"],
        "fps":               fps,
    }


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    rdevice = torch.device(args.retarget_device)

    ts_world, jp_gt, fps, clip_name, obj_rel = load_clip(args.hdf5, args.clip)

    print(f"\nRunning {args.label_a} …")
    pos_a = run_inference(args.ckpt_a, args.stats_a, ts_world, jp_gt, obj_rel,
                          device, args.ode_steps, args.seed)

    pos_b = None
    if args.ckpt_b:
        print(f"\nRunning {args.label_b} …")
        pos_b = run_inference(args.ckpt_b, args.stats_b, ts_world, jp_gt, obj_rel,
                              device, args.ode_steps, args.seed)

    from flow_retarget import build_fk
    fk = build_fk(args.humanoid_xml, rdevice)

    # retarget GT joint positions too (env 2 = green)
    print(f"\nRetargeting GT …")
    pos_gt = jp_gt  # (T, 24, 3) world-frame ground truth

    # order matches PHC env assignment: EEF=blue(0), OBJ=red(1), GT=green(2)
    motion_dict = {}
    motion_dict[f"{clip_name}_{args.label_a}"] = retarget_to_pkl_entry(
        pos_a, fk, rdevice, args.fit_iters, fps, args.label_a)

    if pos_b is not None:
        motion_dict[f"{clip_name}_{args.label_b}"] = retarget_to_pkl_entry(
            pos_b, fk, rdevice, args.fit_iters, fps, args.label_b)

    motion_dict[f"{clip_name}_GT"] = retarget_to_pkl_entry(
        pos_gt, fk, rdevice, args.fit_iters, fps, "GT")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, out_path)
    print(f"\nSaved {len(motion_dict)} clip(s) → {out_path}")

    n_envs = len(motion_dict)
    print(f"\nTrack with PHC:")
    print(f"  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \\")
    print(f"    env=env_im_getup_mcp robot=smpl_humanoid \\")
    print(f"    robot.freeze_hand=True robot.box_body=False \\")
    print(f"    env.z_activation=relu env.obs_v=7 \\")
    print(f"    env.motion_file={out_path} \\")
    print(f"    \"env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']\" \\")
    print(f"    env.num_envs={n_envs} headless=False test=True epoch=-1")


if __name__ == "__main__":
    main()
