"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""
import json
import os
import sys
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterator, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
    DataCollator
)
from olmo_core.data.numpy_dataset import (
    VSLCurriculumType,
    VSLCurriculumConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import (
    get_default_device, 
    seed_all
)
from olmo_core.data.utils import load_array_slice_into_tensor

from olmo_core.distributed.utils import (
    get_rank, 
    all_gather
)

from olmo_eval import HFTokenizer

from tqdm import tqdm
import pandas as pd
from copy import deepcopy 

from ast import literal_eval
from functools import cached_property

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# from metrics import MetricsLogger


@dataclass
class BatchMetricCallback(Callback):
    """
    Adds batch metrics.
    """

    def pre_train(self):
        self.logger = MetricsLogger(self.trainer.save_folder)
        self.batch_metrics = self.logger.load_batch_metrics()
        self.dataset_metrics = self.logger.load_dataset_metrics()

    def post_step(self):
        if get_rank() != 0:
            return
        metrics = self.batch_metrics[self.step]
        for name, value in metrics.items():
            self.trainer.record_metric(f"train/{name}", value)
            if name in self.dataset_metrics:
                value /= self.dataset_metrics[name]
                value = round(value, 4)
                self.trainer.record_metric(f"train/{name}_relative", value)

@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536

def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        fused_ops=False,
        use_flash=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    dataset_config = NumpyDatasetConfig.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/wikipedia_vsl/part*.npy",  # can be globs
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=tokenizer_config,
        work_dir=os.path.join(run_name, "dataset-cache"),
        include_instance_metadata=True,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=64 * 2048, # 256 * 1024,
        seed=0,
        num_workers=4,
        prefetch_factor = 8,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{run_name}",
            rank_microbatch_size=4 * 2048, #16 * 1024,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=5,
            load_key_mapping={
                # For backwards compatibility when loading older checkpoints.
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(
                min_sequence_length=128, warmup_steps=100, enabled=False
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=["arc_easy", "arc_challenge", "openbook_qa", "sciq", "hellaswag", "piqa", "winogrande","commonsense_qa", "trivia_qa_wiki_ppl"],
                tokenizer=tokenizer_config,
                eval_interval=1000,
            ),
        )
        # .with_callback(
        #     "batch_metrics",
        #     BatchMetricCallback()
        # )
    ) 

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    ).merge(overrides)

def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)
    set_random_seeds(config.init_seed)

    device = get_default_device()

    # Build the world mesh, if needed.
    world_mesh = config.model.build_mesh(device=device)

    # # Build components.
    # model = config.model.build(
    #     init_device="meta",
    #     device=device,
    #     max_seq_len=config.dataset.sequence_length,
    #     mesh=world_mesh,
    # )

    # optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, mesh=world_mesh)
    # data_loader.reshuffle(1)
    # batch = None
    # for batch in data_loader:
    #     print(batch)
    #     break
    # trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # # Save config to W&B and each checkpoint dir.
    # config_dict = config.as_config_dict()
    # cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    # cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # # Train.
    # trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]
    debug = any("debug" in o for o in overrides)
    overrides = [o for o in overrides if "debug" not in o]
    prepare_training_environment(debug=debug)
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()
