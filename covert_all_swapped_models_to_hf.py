import os
import sys
from pathlib import Path
import shutil
import subprocess


# Add the src directory to Python path
olmo_core_path = Path.cwd() / "src"
if olmo_core_path.exists():
    sys.path.insert(0, str(olmo_core_path))

print("Current working directory:", os.getcwd())
print(olmo_core_path)

# List of model repository names
model_repos = [
    "olmo2_1B_0.005_32768_0.05_1_random_1_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_random_2_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_last_10_percent_swapping_dict_interval_1",
    "olmo2_1B_0.005_32768_0.05_1_last_10_percent_swapping_dict_interval_100",
    "olmo2_1B_0.005_32768_0.05_1_random_last_20_percent_1_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_random_last_20_percent_2_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_1-13709_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_13710-27418_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_27419-41127_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_41128-54836_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_54837-68545_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_68546-82254_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_82255-95963_swapping_dict",
    "olmo2_1B_0.005_32768_0.05_1_RANDOM_steps_95964-109672_swapping_dict",
]

# Source and target base dirs
SOURCE_BASE = "/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final"
TARGET_BASE = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final"

# Path to bash wrapper script
CONVERT_SH = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/convert_to_hf.sh"


def get_step_dirs(path: str):
    """Return all stepXXXX directories in a repo."""
    return sorted(
        d for d in os.listdir(path)
        if d.startswith("step") and os.path.isdir(os.path.join(path, d))
    )


def main():
    for repo in model_repos:
        src_repo = os.path.join(SOURCE_BASE, repo)
        tgt_repo = os.path.join(TARGET_BASE, repo)

        if not os.path.isdir(src_repo):
            print(f"[WARN] Source repo not found: {src_repo}")
            continue

        # Copy repo if not already in target
        if not os.path.exists(tgt_repo):
            print(f"[COPY] Copying {src_repo} -> {tgt_repo}")
            shutil.copytree(src_repo, tgt_repo)
        else:
            print(f"[SKIP] Repo already exists in target: {tgt_repo}")

        hf_path = os.path.join(tgt_repo, "huggingface")
        os.makedirs(hf_path, exist_ok=True)

        step_dirs = get_step_dirs(tgt_repo)
        hf_step_dirs = get_step_dirs(hf_path)

        missing = sorted(set(step_dirs) - set(hf_step_dirs))

        if not missing:
            print(f"[OK] All checkpoints already converted for {repo}")
            continue

        print(f"[CONVERT] {repo}: {len(missing)} missing step dirs")
        for step in missing:
            in_dir = os.path.join(tgt_repo, step)
            out_dir = os.path.join(hf_path, step)

            print(f"  - Converting {in_dir} -> {out_dir}")
            subprocess.run(
                [CONVERT_SH],
                cwd=os.path.dirname(CONVERT_SH),
                env=dict(os.environ, EXP_DIR=tgt_repo, OUT_BASE=hf_path),
                check=True,
            )
        break

    print("✅ Done copying and converting.")


if __name__ == "__main__":
    main()
