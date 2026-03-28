import sys
import argparse
from itertools import product
import subprocess
import os
import re

python_exe = sys.executable

parser = argparse.ArgumentParser(description="Feature Reliance Experiments")

parser.add_argument("-d", "--datasets", nargs="+", type=str, required=True)
parser.add_argument("-m", "--models", nargs="+", type=str, required=True)
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("-l", "--exp_dir", type=str, default="logs")
parser.add_argument("--cuda-no", type=int, default=0)

args = parser.parse_args()

print("Datasets:", args.datasets)

TEST_SCRIPT_PATH = "test.py"

# ✅ FIXED BASE PARAMETERS
BASE_PARAMETERS = {
    'model.pretrained': [args.pretrained],
    'logging.exp_dir': [args.exp_dir],
    'params.cuda_no': [args.cuda_no],
    'params.seed': [1],
    'dataset.root_path': ['./dataset/tomburgert']   # ✅ FIXED PATH
}

# ✅ KEEP ONLY IMPORTANT TRANSFORMATIONS (simplified)
SETUPS = [
    # No suppression
    {
        'dataaug.test_augmentations': ['resize'],
        'params.protocol_name': ['baseline']
    },

    # Shape removal
    {
        'dataaug.test_augmentations': ['resize_patchshuffle'],
        'dataaug.grid_size': [4],
        'params.protocol_name': ['shape_removed']
    },

    # Texture removal
    {
        'dataaug.test_augmentations': ['resize_bilateral'],
        'dataaug.bilateral_d': [5],
        'dataaug.sigma_color': [75],
        'dataaug.sigma_space': [75],
        'params.protocol_name': ['texture_removed']
    },

    # Color removal
    {
        'dataaug.test_augmentations': ['resize_grayscale'],
        'dataaug.gray_alpha': [1.0],
        'params.protocol_name': ['color_removed']
    }
]

# 🚀 RUN EXPERIMENTS
for dataset in args.datasets:
    for model in args.models:

        log_flag = 'pretrained' if args.pretrained else 'from_scratch'
        base_logging_dir = os.path.join(args.exp_dir, dataset, model, log_flag)
        latest_version = 0
        if os.path.isdir(base_logging_dir):
            candidates = []
            for entry in os.listdir(base_logging_dir):
                match = re.match(r'^version_(\d+)$', entry)
                if match:
                    candidates.append(int(match.group(1)))
            if candidates:
                latest_version = max(candidates)

        BASE_PARAMETERS.update({
            'model.name': [model],
            'params.dataset': [dataset],
            'model.pretrained_version': [latest_version],
        })

        for setup in SETUPS:
            ALL_PARAMS = {**BASE_PARAMETERS, **setup}

            keys = list(ALL_PARAMS.keys())
            values = list(ALL_PARAMS.values())

            for combo in product(*values):
                param_updates = dict(zip(keys, combo))

                print("\nRunning:", param_updates)

                overrides = [f"{k}={v}" for k, v in param_updates.items()]

                subprocess.run([python_exe, TEST_SCRIPT_PATH, *overrides])
