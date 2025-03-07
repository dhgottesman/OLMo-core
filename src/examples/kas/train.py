"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import os
import sys
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional 

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
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

from tqdm import tqdm
import pandas as pd

from ast import literal_eval
from functools import cached_property

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def _get_entities_within_range(data: Dict[str, str], start: int, end: int) -> List[Dict[str, str]]:
    """
    Extracts entities that fall within the given start and end positions.
    
    :param data: Dictionary containing 'entities' as a string-represented list.
    :param start: Start position.
    :param end: End position.
    :return: List of entities within the given range.
    """
    entities = literal_eval(data['entities'])  # Convert string to list of dicts
    return [
        entity for entity in entities
        if start <= entity["entity_token_start"] and end >= entity["entity_token_end"]
    ]

class GlobalBatchProcessor:
    def __init__(self, data_loader, dtype=np.uint32):
        self.data_loader = data_loader
        self.dtype=dtype
        self.process_batches()
        self.metadata = self._get_metadata()
        
    def _get_metadata(self) -> Dict[str, Any]:
        dataset = self.data_loader.dataset

        def _load_metadata(file_path: str) -> list[Dict[str, Any]]:
            """Loads metadata from a compressed CSV file and parses entities."""
            column_names = ['start', 'end', 'id', 'src', 'loc', 'title', 'entities']
            df = pd.read_csv(file_path, names=column_names)
            # df["entities"] = df["entities"].apply(literal_eval)  # Convert string representation to Python objects
            return df.to_dict(orient='records')

        metadata = {
            path: {
                "metadata_path": (metadata_path := os.path.join(os.path.dirname(path), os.path.basename(path).replace(".npy", ".csv.gz"))),
                "metadata": _load_metadata(metadata_path)
            }
            for path in tqdm(dataset.paths)
        }

        return metadata

    @cached_property
    def instance_lengths(self):
        """Gets the chunk lengths."""
        return self.data_loader.dataset.get_instance_lengths()

    def gather_global_batch(self, index_tensor: torch.Tensor) -> torch.Tensor:
        """Gather the 'index' field from all processes and return a global batch tensor."""
        gathered_tensors = all_gather(index_tensor)
        return torch.cat(gathered_tensors, dim=0)
    
    def convert_to_numpy(self, global_batches: List[torch.Tensor]) -> np.ndarray:
        """Convert a list of PyTorch tensors to a NumPy array."""
        return np.concatenate([batch.cpu().numpy() for batch in global_batches], axis=0)
    
    def create_memmap(self, file_path: str, shape: Tuple[int, ...]) -> np.memmap:
        """Create a memory-mapped file."""
        return np.memmap(file_path, dtype=self.dtype, mode='w+', shape=shape)
    
    def write_to_memmap(self, memmap: np.memmap, data: np.ndarray) -> None:
        """Write data to a memory-mapped file."""
        memmap[:] = data[:]
        memmap.flush()
    
    def process_batches(self) -> None:
        """Process data batches and store global batches in a memory-mapped file, tracking example counts."""
        dataset = self.data_loader.dataset
        memmap_file_path = os.path.join(dataset.work_dir, "dataset-common", "dataset-doc-indices.npy")
        batch_sizes_file_path = os.path.join(dataset.work_dir, "dataset-common", "batch-sizes.npy")
        
        # Try to load existing files first
        if os.path.exists(memmap_file_path) and os.path.exists(batch_sizes_file_path):
            if get_rank() == 0:
                print("Loading existing batch sizes and dataset document indices...")
            self.batch_sizes = np.load(batch_sizes_file_path)
            self.dataset_doc_indices = np.memmap(memmap_file_path, dtype=np.uint32, mode='r')
            return
        
        # If files do not exist, process the batches
        self.data_loader.reshuffle(1)
        global_batches = []
        batch_sizes = []
        
        for batch in tqdm(self.data_loader):
            index_tensor = batch['index']
            global_batch = self.gather_global_batch(index_tensor)
            
            if get_rank() == 0:
                global_batches.append(global_batch)
                batch_sizes.append(global_batch.shape[0])
        
        if get_rank() == 0:
            global_batches_np = self.convert_to_numpy(global_batches)
            memmap = self.create_memmap(memmap_file_path, global_batches_np.shape)
            self.write_to_memmap(memmap, global_batches_np)
            np.save(batch_sizes_file_path, np.array(batch_sizes))
            
            # Store the values in the class instance
            self.dataset_doc_indices = memmap
            self.batch_sizes = np.array(batch_sizes)
            
            print(f"Global batches written to {memmap_file_path}")
            print(f"Batch sizes written to {batch_sizes_file_path}")

    def get_batch_doc_indices(self, batch_index: int) -> np.ndarray:
        """
        Given a batch index, compute the offset in dataset_doc_indices and retrieve the correct number of examples.

        Args:
            batch_index (int): The index of the batch to retrieve.
            batch_sizes (np.ndarray): Array containing the number of examples in each batch.
            dataset_doc_indices (np.ndarray): The memory-mapped dataset indices.

        Returns:
            np.ndarray: The subset of dataset_doc_indices corresponding to the given batch.
        """
        if batch_index < 0 or batch_index >= len(self.batch_sizes):
            raise ValueError(f"Invalid batch index {batch_index}. Must be between 0 and {len(self.batch_sizes) - 1}.")

        # Compute the offset by summing batch sizes up to batch_index
        offset = np.sum(self.batch_sizes[:batch_index])

        # Get the number of examples in this batch
        num_examples = self.batch_sizes[batch_index]

        # Retrieve and return the correct slice from dataset_doc_indices
        return self.dataset_doc_indices[offset: offset + num_examples]

    def get_doc_path_offsets_metadata(self, index: int) -> Tuple[str, int, int, Any]:
        """Gets the source path and offsets for a given document."""
        dataset = self.data_loader.dataset

        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(dataset) + index

        # The index of the array within 'self.paths'.
        array_index: Optional[int] = None

        # The index within the corresponding array.
        array_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(dataset.offsets):
            if offset_start <= pos_index < offset_end:
                array_index = i
                array_local_index = pos_index - offset_start
                break

        if array_index is None or array_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(dataset)}")

        path = dataset.paths[array_index]
        indices_path = dataset._get_document_indices_path(path)
        indices = load_array_slice_into_tensor(
            indices_path, array_local_index * 2, array_local_index * 2 + 2, dataset.indices_dtype
        )
        start_idx, end_idx = indices
        start_idx, end_idx = int(start_idx), int(end_idx)

        # Find metadata entry matching start_idx and end_idx
        metadata = next(
            (m for m in self.metadata[path]["metadata"] if start_idx >= m["start"] and end_idx <= m["end"]),
            None
        )

        if metadata is None:
            raise ValueError(f"Unable to find metadata for {self.metadata[path]['metadata_path']}, {start_idx} {end_idx}")

        # Keep only the entities that correspond to this chunk.
        doc_start_idx = metadata["start"]
        # Get the chunk start and end indices offsets relative to the document.
        chunk_start_idx, chunk_end_idx = start_idx - doc_start_idx, end_idx - doc_start_idx
        metadata["entities"] = _get_entities_within_range(metadata, chunk_start_idx, chunk_end_idx)

        # Check that the chunk_token_start_idx, chunk_token_end_idx are correct.
        doc_input_ids = load_array_slice_into_tensor(path, doc_start_idx, metadata["end"], dataset.dtype)
        print(f"CHECK chunk_token_indices {doc_input_ids[chunk_start_idx:chunk_end_idx].tolist()}")
        print("\n")
        print("CHECK REGULAR LOAD", load_array_slice_into_tensor(path, start_idx, end_idx, dataset.dtype))
        return path, start_idx, end_idx, metadata

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
        name=NumpyDatasetType.vsl,
        max_sequence_length=2048,
        min_sequence_length=256,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=1, balanced=True),
        tokenizer=tokenizer_config,
        work_dir=os.path.join(run_name, "dataset-cache"),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=16 * 2048, # 256 * 1024,
        seed=0,
        num_workers=0,
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
                save_interval=100,
                ephemeral_save_interval=50,
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
                tasks=["hellaswag"],
                tokenizer=tokenizer_config,
                eval_interval=100,
            ),
        )
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


    from olmo_eval import HFTokenizer

    tokenizer = config.dataset.tokenizer
    tokenizer = HFTokenizer(
                tokenizer.identifier,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

    processor = GlobalBatchProcessor(data_loader)
    if get_rank() == 0:
        print(processor.get_batch_doc_indices(1), processor.batch_sizes[1], processor.instance_lengths[processor.get_batch_doc_indices(1)[0]])
        # equals = []
        # for doc_idx in processor.get_batch_doc_indices(1):
        #     path, start_idx, end_idx = processor.get_doc_path_and_offsets(doc_idx)
        #     # Try to read from this and compare it to using dataset.__getitem__()
        #     equals.append(
        #         torch.equal(
        #             load_array_slice_into_tensor(path, int(start_idx), int(end_idx), processor.data_loader.dataset.dtype), 
        #             data_loader.dataset.__getitem__(doc_idx)["input_ids"]
        #         )
        #     )
        # print(all(equals))
        doc_idx = processor.get_batch_doc_indices(1)[0]
        print(doc_idx)
        print("\n")
        print(tokenizer.decode(data_loader.dataset.__getitem__(doc_idx)["input_ids"].tolist()))
        print("\n")
        print(data_loader.dataset.__getitem__(doc_idx)["input_ids"].tolist())
        print("\n")        
        print(processor.get_doc_path_offsets_metadata(doc_idx))
        print("\n")        
        print(processor.instance_lengths[doc_idx])

        print("\n\n\n\n-----\n\n\n\n")        

        doc_idx = processor.get_batch_doc_indices(796)[0]
        print(doc_idx)
        print("\n")
        print(tokenizer.decode(data_loader.dataset.__getitem__(doc_idx)["input_ids"].tolist()))
        print("\n")
        print(data_loader.dataset.__getitem__(doc_idx)["input_ids"].tolist())
        print("\n")        
        print(processor.get_doc_path_offsets_metadata(doc_idx))
        print("\n")        
        print(processor.instance_lengths[doc_idx])
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
