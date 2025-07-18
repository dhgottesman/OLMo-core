from pathlib import Path
import sys
import os
import numpy as np
import random
import pickle

# Add the src directory to Python path
olmo_core_path = Path.cwd() / "src"
if olmo_core_path.exists():
    sys.path.insert(0, str(olmo_core_path))

from olmo_core.data.chunk_ordering_utils import (
    get_important_chunks, 
    get_disjoint_entities, 
    sample_injection_points, 
    assign_indices_to_entities,
    shloop)

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

def test_get_important_chunks():

    # Load dataset and compute instance lengths
    dataset = load_dataset("dhgottesman/popqa-kas")
    instance_lengths = np.memmap('/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache/dataset-348b68bc53a9e58ceab6501cae55d803c6b290615d95ac7d98cb0be4a039085d/instance-lengths.npy', mode="r")

    min_chunks = 50
    max_chunks = 100

    important = get_important_chunks(dataset, min_chunks, max_chunks, instance_lengths)
    
    # Extract the original train split
    original_train = dataset['train']

    # Create a lookup from entity_id to example for validation
    subj_id_to_example = {ex['subj_id']: ex for ex in original_train 
                          if min_chunks <= ex['subject_num_chunks'] <= max_chunks}

    for entity in important:
        entity_id = entity['entity_id']
        num_chunks = entity['num_chunks']
        sorted_chunks = entity['chunks']
        sorted_lengths = entity['chunks_lengths']

        # Check chunk count range
        assert min_chunks <= num_chunks <= max_chunks, \
            f"{entity_id} has {num_chunks} chunks, outside the expected range."

        # Check entity exists in filtered dataset
        assert entity_id in subj_id_to_example, \
            f"Entity ID {entity_id} not found in filtered original dataset."

        original_chunks = subj_id_to_example[entity_id]['subject_chunks']

        # Check sorted_chunks is a permutation of original chunks
        assert set(sorted_chunks) == set(original_chunks), \
            f"Chunks mismatch for {entity_id}"

        # Check sorted order by length
        lengths = instance_lengths[sorted_chunks]
        assert all(lengths[i] >= lengths[i+1] for i in range(len(lengths)-1)), \
            f"Chunks not sorted by length for {entity_id}"

    print("All tests passed!")
    return important

def test_all_entities_disjoint(entities):

    disjoint_entities = get_disjoint_entities(entities)
    
    chunks_set = ()

    for entity in disjoint_entities:
        for chunk in entity['chunks']:
            if chunk in chunks_set:
                print("not disjoint")
                break
    print("All entities chunks are disjoint")
    return disjoint_entities

def test_sample_injection_points(num_points_to_sample, max_num_chunks = 100, total_steps = 109672, interval = 5, seed = 0):
    # total_steps = 1000
    # num_points_to_sample = 10
    # max_num_chunks = 5
    # interval = 10
    # seed = 42

    max_valid_start = total_steps - (max_num_chunks - 1) * interval

    sampled = sample_injection_points(
        total_steps,
        num_points_to_sample,
        max_num_chunks,
        interval,
        seed
    )

    # Check length
    assert len(sampled) == num_points_to_sample, \
        f"Expected {num_points_to_sample} points, got {len(sampled)}"

    # Check all points are in range
    assert all(0 <= point < max_valid_start for point in sampled), \
        f"Some points are outside the valid range 0 to {max_valid_start - 1}"

    # Check sorted
    assert sampled == sorted(sampled), "Sampled points are not sorted"

    # Check uniqueness
    assert len(set(sampled)) == len(sampled), "Sampled points are not unique"

    print("All tests passed!")
    return sampled

def test_assign_indices_to_entities(entities, injection_points, interval):
    assigned = assign_indices_to_entities(entities, injection_points, interval)

    # Check each entity
    for entity, start in zip(entities, injection_points):
        entity_id = entity['entity_id']
        num_chunks = entity['num_chunks']
        indices = assigned[entity_id]

        # Correct number of indices
        assert len(indices) == num_chunks, \
            f"{entity_id} should have {num_chunks} indices, got {len(indices)}"

        # Correct spacing
        for i in range(1, len(indices)):
            assert indices[i] - indices[i-1] == interval, \
                f"{entity_id} indices not spaced correctly at position {i}: {indices}"

    print("All tests passed!")
    return assigned

def test_create_swapping_dict(entities, batch_to_chunks_map, all_injection_points_per_entity):

    full_mapping = []
    blacklist = set()

    for entity in entities:
        blacklist.update(entity['chunks'])

    for i, important_chunk in enumerate(entities):
        pts = all_injection_points_per_entity[important_chunk['entity_id']]

        # The 'important_chunk' variable is the integer you need.
        # Pass it directly to your function.
        res = shloop(
            pts,
            important_chunk,
            batch_to_chunks_map,
            blacklist
        )

        # add the chunks already in the injection mapping to a blacklist so they don't get sampled
        blacklist.update(list(set([r[0] for r in res])))

        # extend full mapping with the result
        full_mapping.extend(res)

    swapping_dict = {}
    for key, value in full_mapping:
        swapping_dict[key] = value


    # 1. Check length is 2 * total number of chunks
    total_chunks = sum(len(e['chunks']) for e in entities)
    assert len(swapping_dict) == total_chunks * 2, \
        f"Expected {total_chunks * 2} mappings, got {len(swapping_dict)}"

    # 2. All entity chunks are in keys
    all_entity_chunks = set(chunk for e in entities for chunk in e['chunks'])
    assert all_entity_chunks.issubset(set(swapping_dict.keys())), \
        "Not all entity chunks are included in the swapping dict"

    # 3. Swapped-to chunks should be from batch_to_chunks_map
    all_batch_chunks = {c for batch_chunks in batch_to_chunks_map for c in batch_chunks}

    for src_chunk, target_chunk in swapping_dict.items():
        if src_chunk in all_entity_chunks:
            assert target_chunk in all_batch_chunks, \
                f"{src_chunk} swapped with unexpected chunk {target_chunk}"
        else:
            assert src_chunk in all_batch_chunks and target_chunk in all_entity_chunks, \
                f"Unexpected reverse mapping {src_chunk} -> {target_chunk}"

    # 4. Each used injection point is used to swap at least one chunk
    used_batches = set()
    for batch_id, batch in enumerate(batch_to_chunks_map):
        for chunk in batch:
            if chunk in swapping_dict or swapping_dict.get(chunk) in all_entity_chunks:
                used_batches.add(batch_id)

    expected_batches = set()
    for points in all_injection_points_per_entity.values():
        expected_batches.update(points)

    assert expected_batches.issubset(set(range(len(batch_to_chunks_map)))), \
        "Some assigned injection points not in batch_to_chunks_map"

    assert expected_batches.issubset(used_batches), \
        "Some assigned injection batches not used in swapping"

    print("All tests passed for create_swapping_dict_for_steps_interval!")

def compare_list_of_dicts(list1, list2):
    same = True

    if len(list1) != len(list2):
        print(f"❌ Lists have different lengths: {len(list1)} vs {len(list2)}")
        same = False

    min_len = min(len(list1), len(list2))

    for i in range(min_len):
        dict1 = list1[i]
        dict2 = list2[i]
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            in_1 = key in dict1
            in_2 = key in dict2

            if not in_1:
                print(f"[{i}] Key '{key}' missing in list1")
                same = False
            elif not in_2:
                print(f"[{i}] Key '{key}' missing in list2")
                same = False
            elif dict1[key] != dict2[key]:
                print(f"[{i}] Key '{key}' differs:")
                print(f"  list1[{i}][{key}] = {dict1[key]}")
                print(f"  list2[{i}][{key}] = {dict2[key]}")
                same = False

    if same and len(list1) == len(list2):
        print("✅ Lists of dicts are the same.")

    return same




if __name__ == "__main__":

    # tokenizer_config = TokenizerConfig.dolma2()
    # tokenizer = HFTokenizer(
    #             tokenizer_config.identifier,
    #             pad_token_id=tokenizer_config.pad_token_id,
    #             eos_token_id=tokenizer_config.eos_token_id,
    #             bos_token_id=tokenizer_config.bos_token_id,
    #         )

    # include_instance_metadata = False # Set to true when you want tp retrieve metadata, during training set this to False
    # work_dir = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"

    # dataset_config = NumpyDatasetConfig.glob(
    #     "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",  # can be globs
    #     name=NumpyDatasetType.kas_vsl,
    #     max_sequence_length=2048,
    #     min_sequence_length=64,
    #     vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
    #     tokenizer=tokenizer_config,
    #     work_dir=str(work_dir),
    #     include_instance_metadata=include_instance_metadata,
    # )

    batch_to_chunks_map = np.load("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/batch_indices.npy", allow_pickle=True)
    # kas_dataset = dataset_config.build()

    # important_chunks = test_get_important_chunks()
    # test_all_entities_disjoint(important_chunks)


    with open('/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/chunk_ordering_data/rare_disjoint_entities_50-100.pkl', 'rb') as f:
        rare_entities = pickle.load(f)


    injection_points = test_sample_injection_points(len(rare_entities), interval=10)
    batch_per_entity = test_assign_indices_to_entities(rare_entities, injection_points, interval=10)

    test_create_swapping_dict(rare_entities, batch_to_chunks_map, batch_per_entity)





