import os
import json
import torch

from .numpy_dataset import (
    NumpyDatasetConfig,
    NumpyDatasetType,
    VSLCurriculumType,
    VSLCurriculumConfig,
)

from .data_loader import NumpyDataLoaderConfig
from .tokenizer import TokenizerConfig
from .collator import DataCollator

# Set your new cache base directory (change this to your preferred location)
cache_base = "/home/joberant/NLP_2425b/shirab6"

# Set all relevant Hugging Face cache directories
os.environ["HF_HOME"] = cache_base
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["HF_TOKENIZERS_CACHE"] = os.path.join(cache_base, "tokenizers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_dataloader_batches(dataloader, output_path='batches.json'):
    batches_dict = {}

    for i, batch in enumerate(dataloader):
        # Convert the batch to JSON-serializable format
        if isinstance(batch, (list, tuple)):
            batch_serializable = [tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
        elif isinstance(batch, dict):
            batch_serializable = {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:  # Single tensor
            batch_serializable = batch.cpu().tolist() if isinstance(batch, torch.Tensor) else batch

        batches_dict[i] = batch_serializable

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(batches_dict, f)

    print(f"Saved {len(batches_dict)} batches to {output_path}")

if __name__ == "__main__":

    tokenizer_config = TokenizerConfig.dolma2()
    tokenizer = HFTokenizer(
                tokenizer_config.identifier,
                pad_token_id=tokenizer_config.pad_token_id,
                eos_token_id=tokenizer_config.eos_token_id,
                bos_token_id=tokenizer_config.bos_token_id,
            )

    include_instance_metadata = False # Set to true when you want tp retrieve metadata, during training set this to False
    work_dir = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"

    dataset_config = NumpyDatasetConfig.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",  # can be globs
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        include_instance_metadata=include_instance_metadata,
    )
    dataset = dataset_config.build()

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=32768,
        seed=0,
        num_workers=8,
        prefetch_factor = 16,
    )

    dataloader = data_loader_config.build(dataset)
    dataloader.reshuffle(1)

