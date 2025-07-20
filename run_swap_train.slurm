#! /bin/sh
#SBATCH --job-name=train_olmo2_1B_swap_100_epochs
#SBATCH --output=train_olmo2_1B_swap_100_epochs.out
#SBATCH --error=train_olmo2_1B_swap_100_epochs.err
#SBATCH --time=1-00:00:00
#SBATCH --gpus=4               # Request 4 GPU
#SBATCH --nodes=1
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01

nvidia-smi

WANDB_API_KEY="507c1b87f60d6922a9ea243589deb86f3bce5c0f"

nohup torchrun --nproc_per_node=4 --master_port=29501 src/examples/kas/train_with_swaps.py hp_final olmo2_1B 1 0.005 32768 0.05 8192 100 &

wait
