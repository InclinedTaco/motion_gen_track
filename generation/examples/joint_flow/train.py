"""
Train the joint-flow model.

Usage (from repo root):
    python examples/joint_flow/train.py \
        --hdf5  data/amass/hdf5/accad_research_filtered_taskspace.h5 \
        --outdir checkpoints/joint_flow

Key hyper-parameters can be overridden via CLI flags (see argparse below).
"""

import argparse
import sys
import time
from pathlib import Path

# make flow_matching importable when run from repo root or examples/joint_flow
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flow_matching.path import CondOTProbPath

from dataset import JointFlowDataset
from model import JointFlowMLP, CVAEEncoder, CVAEPrior, reparameterize, kl_divergence


# ---------------------------------------------------------------------------

def split_clips_for_validation(ds: JointFlowDataset, val_frac: float, seed: int = 0):
    n_clips = ds.num_clips
    if n_clips < 2:
        raise ValueError("Need at least two clips for clip-level train/val split")

    clip_lengths = [ds.clip_len(i) for i in range(n_clips)]
    target_val_frames = max(1, int(sum(clip_lengths) * val_frac))
    perm = torch.randperm(n_clips, generator=torch.Generator().manual_seed(seed)).tolist()

    val_clips = []
    val_frames = 0
    for clip_idx in perm:
        if val_frames >= target_val_frames and val_clips:
            break
        val_clips.append(clip_idx)
        val_frames += clip_lengths[clip_idx]
        if len(val_clips) == n_clips - 1:
            break

    val_clip_set = set(val_clips)
    train_clips = [i for i in range(n_clips) if i not in val_clip_set]
    if not train_clips:
        moved_clip = val_clips.pop()
        train_clips = [moved_clip]

    return train_clips, val_clips


def repeated_train_frame_indices(
    ds: JointFlowDataset, train_clips: list[int], oversample_factors: list[int]
) -> list[int]:
    repeated_indices = []
    for clip_idx in train_clips:
        factor = oversample_factors[ds.clip_source_id(clip_idx)]
        clip_indices = ds.frame_indices_for_clips([clip_idx])
        repeated_indices.extend(clip_indices * factor)
    return repeated_indices


def validation_frame_indices(ds: JointFlowDataset, val_clips: list[int]) -> list[int]:
    return ds.frame_indices_for_clips(val_clips)


def generate_prev_frame(model, prev_cond, joint_dim, n_steps):
    """Roll the velocity model from noise to a pose (normalized joint space) via
    midpoint integration, for scheduled sampling. Caller wraps this in no_grad;
    the result is fed back in as the prev-frame, i.e. detached from the graph."""
    x = torch.randn(prev_cond.shape[0], joint_dim, device=prev_cond.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=prev_cond.device)
        k1 = model(x, t, prev_cond)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt, prev_cond)
        x = x + dt * k2
    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",       required=True, nargs="+", help="One or more taskspace HDF5 files")
    p.add_argument("--oversample", type=int,      nargs="+", default=None,
                   help="Repeat factor per HDF5 file (parallel to --hdf5); e.g. --oversample 1 1 33")
    p.add_argument("--outdir",  default="checkpoints/joint_flow")
    p.add_argument("--loss-log", default=None, help="path to txt file for per-epoch train/val loss (default: <outdir>/loss_log.txt)")
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--batch",   type=int,   default=32768)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--hidden",  type=int,   default=512)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--window",  type=int,   default=10,  help="sliding window size K")
    p.add_argument("--no-eef",         action="store_true", help="drop EEF from task-space; condition on root_vel+pelvis_z only (3d/frame)")
    p.add_argument("--use-6d",         action="store_true", help="output 6D rotation per body joint (138D) instead of joint positions (69D); skips retargeter at inference")
    p.add_argument("--predict-root",   action="store_true", help="(requires --use-6d) also predict pelvis orientation in 6D (144D total); eliminates velocity heuristic at inference")
    p.add_argument("--use-prev-frame", action="store_true", help="condition on previous predicted frame")
    p.add_argument("--prev-dropout",   type=float, default=0.15)
    p.add_argument("--prev-noise",     type=float, default=0.1)
    p.add_argument("--scheduled-sampling", action="store_true",
                   help="autoregressive training: feed the model's own one-step prediction as the prev-frame with ramping probability (requires --use-prev-frame)")
    p.add_argument("--ss-prob",  type=float, default=0.5, help="final scheduled-sampling replacement probability")
    p.add_argument("--ss-ramp",  type=int,   default=100, help="epochs to linearly ramp scheduled-sampling probability 0 -> --ss-prob")
    p.add_argument("--ss-steps", type=int,   default=5,   help="ODE steps used to generate the prev-frame during scheduled sampling")
    p.add_argument("--use-obj",         action="store_true", help="append obj_com_pos_rel (3D) to task-space condition; uses (2,2,2) sentinel for non-OMOMO clips")
    p.add_argument("--use-task-goals",  action="store_true", help="append pickup/place/desired-hold object goals (9D); derived from OMOMO object trajectories")
    p.add_argument("--motion-frame", choices=["world", "start"], default="world",
                   help="frame for root_vel_xy conditioning: world axes or frame-0 heading axes")
    p.add_argument("--motion-repr", choices=["velocity", "future_dxdy"], default="velocity",
                   help="locomotion conditioning representation for XY motion")
    p.add_argument("--motion-horizon-s", type=float, default=0.5,
                   help="future horizon in seconds when --motion-repr future_dxdy is used")
    p.add_argument("--task-goal-frame", choices=["current", "start"], default="current",
                   help="coordinate frame for pickup/place goals when --use-task-goals is enabled")
    p.add_argument("--use-carry-phase", action="store_true", help="append 1D carry-phase indicator; non-object clips get zero")
    p.add_argument("--use-task-phase",  action="store_true", help="append task phase one-hot")
    p.add_argument("--task-phase-schema", choices=["4", "5"], default="4",
                   help="task phase schema when --use-task-phase is enabled: 4=approach/pickup/carry_walk/place, 5=approach/pickup/carry_stand/carry_walk/place")
    p.add_argument("--use-cvae",       action="store_true", help="learn latent task-space via CVAE")
    p.add_argument("--latent-dim",     type=int,   default=64,  help="CVAE latent dimension")
    p.add_argument("--kl-weight",      type=float, default=1e-3, help="weight on KL loss")
    p.add_argument("--kl-anneal",      type=int,   default=50,  help="epochs to linearly anneal KL from 0")
    p.add_argument("--val-frac",    type=float, default=0.05)
    p.add_argument("--save-every",  type=int,   default=20)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    loss_log_path = Path(args.loss_log) if args.loss_log else out / "loss_log.txt"
    loss_log_f = open(loss_log_path, "w")
    loss_log_f.write("epoch\ttrain_loss\tval_loss\n")
    loss_log_f.flush()

    # --- data ---
    use_prev     = args.use_prev_frame
    use_cvae     = args.use_cvae
    use_6d       = args.use_6d
    no_eef       = args.no_eef
    use_obj      = args.use_obj
    use_task_goals = args.use_task_goals
    motion_frame = args.motion_frame
    motion_repr = args.motion_repr
    motion_horizon_s = args.motion_horizon_s
    use_carry_phase = args.use_carry_phase
    use_task_phase = args.use_task_phase
    task_phase_dim = int(args.task_phase_schema)
    predict_root = args.predict_root and use_6d
    joint_dim    = (144 if predict_root else 138) if use_6d else 69
    if args.scheduled_sampling and not use_prev:
        raise ValueError("--scheduled-sampling requires --use-prev-frame")
    if args.scheduled_sampling and use_cvae:
        raise ValueError("--scheduled-sampling is not supported with --use-cvae")
    prev_dim  = joint_dim if use_prev else 0
    ts_dim    = (
        (3 if no_eef else 9)
        + (3 if use_obj else 0)
        + (9 if use_task_goals else 0)
        + (1 if use_carry_phase else 0)
        + (task_phase_dim if use_task_phase else 0)
    )
    factors = args.oversample or [1] * len(args.hdf5)
    if len(factors) != len(args.hdf5):
        raise ValueError(f"--oversample has {len(factors)} values but --hdf5 has {len(args.hdf5)}")
    print(
        f"Loading dataset (window={args.window}, use_prev_frame={use_prev}, "
        f"no_eef={no_eef}, use_6d={use_6d}, use_obj={use_obj}, "
        f"use_task_goals={use_task_goals}, motion_frame={motion_frame}, "
        f"motion_repr={motion_repr}, motion_horizon_s={motion_horizon_s}, "
        f"task_goal_frame={args.task_goal_frame}, "
        f"use_carry_phase={use_carry_phase}, use_task_phase={use_task_phase}, "
        f"task_phase_schema={task_phase_dim})"
    )
    for p, n in zip(args.hdf5, factors):
        print(f"  x{n}  {p}")
    full_ds = JointFlowDataset(
        args.hdf5,
        window=args.window,
        use_prev_frame=use_prev,
        no_eef=no_eef,
        use_6d=use_6d,
        predict_root=predict_root,
        use_obj=use_obj,
        use_task_goals=use_task_goals,
        motion_frame=motion_frame,
        motion_repr=motion_repr,
        motion_horizon_s=motion_horizon_s,
        task_goal_frame=args.task_goal_frame,
        use_carry_phase=use_carry_phase,
        use_task_phase=use_task_phase,
        task_phase_dim=task_phase_dim,
        scheduled_sampling=args.scheduled_sampling,
    )
    full_ds.save_stats(str(out / "stats.json"))
    if args.scheduled_sampling:
        print(f"  scheduled sampling ON: ss_prob={args.ss_prob} ramp={args.ss_ramp}ep ss_steps={args.ss_steps}")
    cond_dim = prev_dim + args.window * ts_dim
    flow_cond_dim = args.latent_dim if use_cvae else cond_dim
    print(f"  {len(full_ds):,} frames  |  cond_dim={cond_dim}  flow_cond_dim={flow_cond_dim}  joint_dim={joint_dim}")

    train_clips, val_clips = split_clips_for_validation(full_ds, args.val_frac, seed=0)
    train_indices = repeated_train_frame_indices(full_ds, train_clips, factors)
    val_indices = validation_frame_indices(full_ds, val_clips)
    n_train = len(train_indices)
    n_val = len(val_indices)
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    train_unique_frames = sum(full_ds.clip_len(i) for i in train_clips)
    print(
        f"  split: train_clips={len(train_clips)} val_clips={len(val_clips)} "
        f"| train_frames={n_train:,} ({train_unique_frames:,} unique) val_frames={n_val:,}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    # --- models ---
    model = JointFlowMLP(
        joint_dim=joint_dim,
        cond_dim=flow_cond_dim,
        hidden_size=args.hidden,
        n_layers=args.layers,
    ).to(device)

    all_params = list(model.parameters())

    if use_cvae:
        encoder = CVAEEncoder(cond_dim, 69, args.latent_dim).to(device)
        prior   = CVAEPrior(cond_dim, args.latent_dim).to(device)
        all_params += list(encoder.parameters()) + list(prior.parameters())
    else:
        encoder = prior = None

    n_params = sum(p.numel() for p in all_params)
    print(f"  Total params: {n_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    prob_path = CondOTProbPath()
    mse       = nn.MSELoss()

    # --- training loop ---
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if use_cvae:
            encoder.train(); prior.train()
        train_flow_loss = 0.0
        train_kl_loss   = 0.0

        kl_w = min(1.0, epoch / max(1, args.kl_anneal)) * args.kl_weight
        ss_p = (
            min(1.0, epoch / max(1, args.ss_ramp)) * args.ss_prob
            if args.scheduled_sampling
            else 0.0
        )

        for batch in train_loader:
            cond = batch[0].to(device)
            x1   = batch[-1].to(device)

            if use_prev:
                if args.prev_noise > 0:
                    cond[:, :prev_dim] += torch.randn(cond.shape[0], prev_dim, device=device) * args.prev_noise
                if args.prev_dropout > 0:
                    mask = torch.rand(cond.shape[0], device=device) < args.prev_dropout
                    cond[mask, :prev_dim] = 0.0
                if ss_p > 0:
                    # scheduled sampling: overwrite a fraction of prev-frames with
                    # the model's own one-step prediction (structured error that
                    # random noise/dropout above cannot reproduce).
                    ss_mask = torch.rand(cond.shape[0], device=device) < ss_p
                    if ss_mask.any():
                        prev_cond = batch[1][ss_mask.cpu()].to(device)
                        with torch.no_grad():
                            x_hat = generate_prev_frame(model, prev_cond, joint_dim, args.ss_steps)
                        cond[ss_mask, :prev_dim] = x_hat

            if use_cvae:
                mu_enc, logvar_enc     = encoder(cond, x1)
                mu_prior, logvar_prior = prior(cond)
                z         = reparameterize(mu_enc, logvar_enc)
                kl        = kl_divergence(mu_enc, logvar_enc, mu_prior, logvar_prior)
                condition = z
            else:
                condition = cond
                kl        = torch.tensor(0.0)

            x0 = torch.randn_like(x1)
            t  = torch.rand(x1.shape[0], device=device)

            path_sample = prob_path.sample(x_0=x0, x_1=x1, t=t)
            flow_loss   = mse(model(path_sample.x_t, path_sample.t, condition), path_sample.dx_t)
            loss        = flow_loss + kl_w * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            train_flow_loss += flow_loss.item() * len(cond)
            train_kl_loss   += kl.item()         * len(cond)

        train_flow_loss /= n_train
        train_kl_loss   /= n_train
        scheduler.step()

        # --- validation (teacher-forced, use prior not encoder) ---
        model.eval()
        if use_cvae:
            encoder.eval(); prior.eval()
        val_flow_loss = 0.0
        val_kl_loss   = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cond = batch[0].to(device)
                x1   = batch[-1].to(device)
                if use_cvae:
                    mu_enc, logvar_enc     = encoder(cond, x1)
                    mu_prior, logvar_prior = prior(cond)
                    condition = mu_prior
                    val_kl_loss += kl_divergence(mu_enc, logvar_enc, mu_prior, logvar_prior).item() * len(cond)
                else:
                    condition = cond
                x0 = torch.randn_like(x1)
                t  = torch.rand(x1.shape[0], device=device)
                path_sample = prob_path.sample(x_0=x0, x_1=x1, t=t)
                val_flow_loss += mse(model(path_sample.x_t, path_sample.t, condition), path_sample.dx_t).item() * len(cond)
        val_flow_loss /= n_val
        val_kl_loss   /= n_val

        elapsed = time.time() - t0

        save_dict = {"epoch": epoch, "model": model.state_dict(),
                     "val_loss": val_flow_loss, "args": vars(args)}
        if use_cvae:
            save_dict["encoder"] = encoder.state_dict()
            save_dict["prior"]   = prior.state_dict()

        if val_flow_loss < best_val:
            best_val = val_flow_loss
            torch.save(save_dict, str(out / "best.pt"))



        # --- loss log ---
        loss_log_f.write(f"{epoch}\t{train_flow_loss:.6f}\t{val_flow_loss:.6f}\n")
        loss_log_f.flush()

        # --- per-epoch log line ---
        if use_cvae:
            print(f"epoch {epoch:4d}/{args.epochs}"
                  f"  train={train_flow_loss:.4f}  kl={train_kl_loss:.4f}  kl_w={kl_w:.1e}"
                  f"  val={val_flow_loss:.4f}  best={best_val:.4f}"
                  f"  ({elapsed:.1f}s)", flush=True)
        else:
            print(f"epoch {epoch:4d}/{args.epochs}"
                  f"  train={train_flow_loss:.4f}  val={val_flow_loss:.4f}  best={best_val:.4f}"
                  f"  ({elapsed:.1f}s)", flush=True)

    loss_log_f.close()
    print(f"\nDone.  best_val={best_val:.5f}  →  {out}/")


if __name__ == "__main__":
    main()
