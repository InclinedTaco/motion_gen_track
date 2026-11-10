#!/usr/bin/env python3
"""Portable entry points for the generation → PHC integration."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "generation"
TRACKING = ROOT / "tracking"


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def absolute(value):
    return str(Path(value).expanduser().resolve())


def hydra_string(value):
    # Hydra parses each argv element again, including paths with spaces/commas.
    return json.dumps(str(value))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check dependencies without importing the simulator")
    doctor.add_argument("--tracking", action="store_true")
    for name in ("generate", "track"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True, type=absolute, help="Flow model best.pt")
        sub.add_argument("--stats", required=True, type=absolute, help="Matching stats.json")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--steps", type=positive, default=10, help="ODE solver steps")
        sub.add_argument("--stride", type=positive, default=5)
        sub.add_argument("--dry-run", action="store_true", help="Print the command; do not load models")
        if name == "generate":
            sub.add_argument("--motion", choices=["stand", "constant_walk", "curved_walk", "stop_start_walk"], default="constant_walk")
            sub.add_argument("--seconds", type=positive, default=5)
            sub.add_argument("--device", default="cpu")
            sub.add_argument("--out", type=absolute, default=str(ROOT / "outputs/generated.gif"))
        else:
            sub.add_argument("--composer", type=absolute, default=str(TRACKING / "output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth"))
            sub.add_argument("--primitives", type=absolute, default=str(TRACKING / "output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth"))
            sub.add_argument("--standing-motion", type=absolute, default=str(TRACKING / "sample_data/amass_isaac_standing_upright_slim.pkl"))
            sub.add_argument("--feedback", choices=["robot", "prediction"], default="robot")
            sub.add_argument("--envs", type=positive, default=1)
            sub.add_argument("--episode-steps", type=positive, default=300)
            sub.add_argument("--headless", action="store_true")
    return p


def build_command(args):
    if args.command == "generate":
        return GENERATION, [sys.executable, "examples/joint_flow/visualization/animate_horizon_x15.py",
            "--checkpoint", args.checkpoint, "--stats", args.stats, "--motion", args.motion,
            "--duration", str(args.seconds), "--steps", str(args.steps), "--stride", str(args.stride),
            "--seed", str(args.seed), "--device", args.device, "--out", args.out]
    return TRACKING, [sys.executable, "phc/run_hydra.py", "learning=im_mcp",
        "exp_name=motion_generation_tracking", "env=env_im_getup_mcp",
        "env.task=HumanoidImMCPDemo", "robot=smpl_humanoid",
        "robot.freeze_hand=True", "robot.box_body=False", "env.z_activation=relu", "env.obs_v=7",
        "env.getup_schedule=False", "env.zero_out_far=False", "env.power_reward=False",
        "env.motion_file=" + hydra_string(args.standing_motion),
        "env.models=[" + hydra_string(args.primitives) + "]",
        "learning.params.load_checkpoint=True", "learning.params.load_path=" + hydra_string(args.composer),
        "learning.params.network.has_softmax=True", "epoch=0", "test=True", "no_log=True",
        "headless=" + str(args.headless), "env.num_envs=" + str(args.envs),
        "env.episode_length=" + str(args.episode_steps), "seed=" + str(args.seed),
        "+env.flow_ref_mode=inprocess", "+env.flow_ckpt=" + hydra_string(args.checkpoint),
        "+env.flow_stats=" + hydra_string(args.stats), "+env.flow_feedback_source=" + args.feedback,
        "+env.flow_reset_history_on_env_reset=True", "+env.flow_stand_steps=90",
        "+env.flow_hold_stand_ref=False", "+env.flow_seed=" + str(args.seed),
        "+env.flow_stride=" + str(args.stride), "+env.flow_ode_steps=" + str(args.steps)]


def validate_assets(args):
    names = ["checkpoint", "stats"]
    if args.command == "track":
        names += ["composer", "primitives", "standing_motion"]
    missing = [getattr(args, key) for key in names if not Path(getattr(args, key)).is_file()]
    if missing:
        raise ValueError("Missing assets (see docs/setup.md):\n  " + "\n  ".join(missing))
    with open(args.stats) as handle:
        stats = json.load(handle)
    if args.stride > stats.get("horizon", 0):
        raise ValueError("--stride must not exceed the checkpoint's horizon")
    # The online adapter decodes 76-value humanoid states. X17 object_place
    # checkpoints use 85 values and need a different adapter.
    horizon = stats["horizon"]
    if stats.get("condition_schema", "phase") != "phase" or len(stats.get("target_mean", [])) != horizon * 76:
        raise ValueError("This entry point requires a 76D phase-conditioned X16 checkpoint; see docs/architecture.md")


def doctor(tracking=False):
    modules = ["torch", "torchdiffeq", "numpy", "scipy", "matplotlib", "h5py"]
    if tracking:
        modules += ["isaacgym", "hydra", "omegaconf", "rl_games", "smpl_sim", "flask"]
    missing = []
    for name in modules:
        found = importlib.util.find_spec(name) is not None
        print(("OK      " if found else "MISSING ") + name)
        if not found:
            missing.append(name)
    print("Dependency discovery only; this does not validate GPU binaries, assets, or tracking.")
    return int(bool(missing))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.tracking)
    cwd, command = build_command(args)
    print("Working directory:", cwd, flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        validate_assets(args)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if args.command == "generate":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(GENERATION), str(TRACKING), env.get("PYTHONPATH", "")])
    # All asset paths were resolved before changing cwd. Do not load torch here:
    # Isaac Gym requires its imports to precede torch in the child process.
    return subprocess.call(command, cwd=cwd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
