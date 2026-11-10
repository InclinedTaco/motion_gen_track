"""Filter out highly dynamic motions (jumps, hops, leaps, crawls, etc.) from a PHC pkl."""
import sys
import argparse
import joblib
import numpy as np

EXCLUDE_KEYWORDS = [
    "jump", "hop", "leap", "skip", "cart", "flip",
    "crawl", "crouch", "lie", "roll", "martial",
    "run", "sprint", "jog",
]

def is_dynamic_by_height(entry, threshold=0.3):
    """Exclude if root height varies more than threshold (jumps go high)."""
    trans = entry.get('root_trans_offset', entry.get('trans_orig', None))
    if trans is None:
        return False
    if hasattr(trans, 'numpy'):
        trans = trans.numpy()
    z = trans[:, 2]
    return (z.max() - z.min()) > threshold

def has_no_movement(entry, max_joint_var=0.02):
    """Exclude if joints barely change over time — static/calibration poses."""
    pose = entry.get('pose_quat_global', None)
    if pose is None:
        return False
    if hasattr(pose, 'numpy'): pose = pose.numpy()
    joint_var = float(pose.var(axis=0).mean())
    return joint_var < max_joint_var

# KIT has named files — keep only sequences matching these locomotion keywords.
KIT_LOCOMOTION_INCLUDE = [
    "walk", "run", "turn", "straight",
    "walkinclockwise", "walkincounterclockwise",
    "clockwise", "backward", "forward", "trial_",
    "push_recovery", "displace",
]

parser = argparse.ArgumentParser()
parser.add_argument("in_path", nargs="?", default="data/amass/pkls/accad_isaac_upright_slim.pkl")
parser.add_argument("out_path", nargs="?", default="data/amass/pkls/accad_isaac_upright_slim_filtered.pkl")
parser.add_argument("--no-keyword-filter", action="store_true", help="Skip name-based keyword exclusion")
parser.add_argument("--kit-locomotion-only", action="store_true", help="For KIT: keep only sequences matching locomotion keywords")
args = parser.parse_args()

data = joblib.load(args.in_path)
filtered = {}
excluded = []

for key, entry in data.items():
    name = key.lower()
    if args.kit_locomotion_only and not any(kw in name for kw in KIT_LOCOMOTION_INCLUDE):
        excluded.append(key)
        continue
    if not args.no_keyword_filter and not args.kit_locomotion_only and any(kw in name for kw in EXCLUDE_KEYWORDS):
        excluded.append(key)
        continue
    if is_dynamic_by_height(entry, threshold=0.3):
        excluded.append(key)
        continue
    if has_no_movement(entry):
        excluded.append(key)
        continue
    filtered[key] = entry

print(f"Total: {len(data)}  Kept: {len(filtered)}  Excluded: {len(excluded)}")
print("\nExcluded motions:")
for k in excluded:
    print(" ", k)

joblib.dump(filtered, args.out_path)
print(f"\nSaved to {args.out_path}")
