import os
import sys
from pathlib import Path
import pickle

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
from datasets import load_dataset

if __name__ == "__main__":

    # load the swapping dicts

    interval = 1 # change to load a different swapping dict intervals are 1, 50, 100
    with open('/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/chunk_ordering_data/swapping_dict_interval_{interval}.pkl', 'rb') as f:
        swapping_dict = pickle.load(f)


    # Prepare the dataset
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
        swapping_dict = swapping_dict,
    )   

    reordered_dataset = dataset_config.build()  

    # Prepare the dataloader
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=32768,
        seed=0,
        num_workers=8,
        prefetch_factor = 16,
    )   

    dataloader = data_loader_config.build(reordered_dataset)
    dataloader.reshuffle(1) 

    # train!