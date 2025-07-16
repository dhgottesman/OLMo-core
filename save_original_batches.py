import pickle
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


# Add the src directory to Python path
olmo_core_path = Path.cwd() / "src"
if olmo_core_path.exists():
    sys.path.insert(0, str(olmo_core_path))

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

# Set your new cache base directory (change this to your preferred location)
cache_base = "/home/joberant/NLP_2425b/shirab6"

# Set all relevant Hugging Face cache directories
os.environ["HF_HOME"] = cache_base
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["HF_TOKENIZERS_CACHE"] = os.path.join(cache_base, "tokenizers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from olmo_eval import HFTokenizer
def save_checkpoint(processed_batches, all_indices, checkpoint_file="/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/batch_indices_checkpoint.pkl"):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'processed_batches': processed_batches,
        'all_indices': all_indices
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_file="/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/batch_indices.pkl"):
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return {'processed_batches': 0, 'all_indices': []}

def save_batch_indices(dataloader, output_file='/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/batch_indices.pkl'):
    """
    Iterate through a DataLoader and save all batch['index'] values to a .pkl file.
    
    Args:
        dataloader: DataLoader object
        output_file: Path to save the .pkl file
    """
    # Set up checkpoint file path
    checkpoint_file = output_file.replace('.pkl', '_checkpoint.pkl')
    
    # Load existing checkpoint if available
    checkpoint_data = load_checkpoint(checkpoint_file)
    start_idx = checkpoint_data['processed_batches']
    all_indices = checkpoint_data['all_indices']
    
    if start_idx > 0:
        print(f"Resuming from batch {start_idx} (found existing checkpoint)")
    
    try:
        # Iterate through the dataloader
        for i, batch in enumerate(tqdm(dataloader, desc="Saving batch indices", initial=start_idx)):
            
            # Skip batches we've already processed
            if i < start_idx:
                continue
                
            batch_indices = batch['index']
            
            # Convert to numpy if it's a tensor
            if torch.is_tensor(batch_indices):
                batch_indices = batch_indices.numpy()
            
            all_indices.append(batch_indices)
            
            # Save checkpoint every 1000 batches to minimize I/O overhead
            if (i + 1) % 1000 == 0:
                save_checkpoint(i + 1, all_indices, checkpoint_file)
                print(f"Checkpoint saved at batch {i + 1}")
        
        # Save to .pkl file
        with open(output_file, 'wb') as f:
            pickle.dump(all_indices, f)
        
        print(f"Saved {len(all_indices)} indices to {output_file}")
        
        # Clean up checkpoint file on successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Checkpoint file cleaned up")
        
        return all_indices
        
    except Exception as e:
        # Save checkpoint before crashing
        current_batch = len(all_indices) + start_idx
        save_checkpoint(current_batch, all_indices, checkpoint_file)
        print(f"Error occurred at batch {current_batch}. Checkpoint saved.")
        print(f"To resume, simply run the script again.")
        raise

if __name__ == "__main__":

    print("Loading dataset")
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

    print("Preparing dataloader")
    data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=32768,
    seed=0,
    num_workers=4,
    prefetch_factor = 16,
    )

    dataloader = data_loader_config.build(dataset)
    dataloader.reshuffle(1)

    print("Iterating over the batches")
    save_batch_indices(dataloader)