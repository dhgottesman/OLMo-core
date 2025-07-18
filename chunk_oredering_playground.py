import os
import sys
from pathlib import Path
import random
import numpy as np
import pickle
from typing import List

# Add the src directory to Python path
import sys
src_path = "/home/joberant/NLP_2425b/vishnevsky/knowledge-analysis-suite/OLMo-core/src"
#src_path = "/home/joberant/NLP_2425b/vishnevsky/knowledge-analysis-suite/OLMo-core/hp_final"
os.chdir(src_path)
sys.path.insert(0, src_path)

print("Current working directory:", os.getcwd())
print("olmo_core should now be importable")


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
#cache_base = "/home/joberant/NLP_2425b/yoavbaron"
cache_base = "/home/joberant/NLP_2425b/vishnevsky/.cache/huggingface"

# Set all relevant Hugging Face cache directories
os.environ["HF_HOME"] = cache_base
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["HF_TOKENIZERS_CACHE"] = os.path.join(cache_base, "tokenizers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from olmo_eval import HFTokenizer
from datasets import load_dataset

tokenizer_config = TokenizerConfig.dolma2()
tokenizer = HFTokenizer(
            tokenizer_config.identifier,
            pad_token_id=tokenizer_config.pad_token_id,
            eos_token_id=tokenizer_config.eos_token_id,
            bos_token_id=tokenizer_config.bos_token_id,
        )

include_instance_metadata = False # Set to true when you want tp retrieve metadata, during training set this to False
work_dir = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"

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
# kas_dataset = dataset_config.build()

# data_loader_config = NumpyDataLoaderConfig(
#     global_batch_size=32768,
#     seed=0,
#     num_workers=8,
#     prefetch_factor = 16,
# )

# dataloader = data_loader_config.build(kas_dataset)
# dataloader.reshuffle(1)

# def get_important_chunks(dataset, min_num_chunks, max_num_chunks, instance_lengths):
#     # Filter the dataset
#     filtered_dataset = dataset['train'].filter(
#         lambda example: min_num_chunks <= example['subject_num_chunks'] <= max_num_chunks
#     )   

#     # Create list of dictionaries with subject info and chunk lengths
#     result_list = []    

#     for example in filtered_dataset:
        
#         subject_name = example['subj']
#         subject_id = example['subj_id']
#         chunks = example['subject_chunks']
#         num_chunks = example['subject_num_chunks']

#         chunk_lengths = instance_lengths[chunks]

#         # if subject_name == 'Madison':
#         #     print(chunks)

#         # Sort chunks by their lengths (descending order)
#         if len(chunk_lengths) > 0:
#             # Create pairs of (chunk, length) and sort by length
#             chunk_length_pairs = list(zip(chunks, chunk_lengths))
#             chunk_length_pairs.sort(key=lambda x: x[1], reverse=True)

#             # Separate back into sorted chunks and lengths
#             sorted_chunks = [pair[0] for pair in chunk_length_pairs]
#             sorted_lengths = [pair[1] for pair in chunk_length_pairs]
#         else:
#             sorted_chunks = chunks
#             sorted_lengths = chunk_lengths

#         subject_dict = {
#             'entity_id': subject_id,
#             'entity_name': subject_name,
#             'num_chunks': num_chunks,
#             'chunks': sorted_chunks,
#             'chunks_lengths': sorted_lengths
#         }
        
#         result_list.append(subject_dict)    

#     # Sort the list by number of chunks (descending order)
#     result_list.sort(key=lambda x: x['num_chunks'], reverse=True)

#     return result_list


# ds = load_dataset("dhgottesman/popqa-kas")

# """
# importsnt chunks has the following structure:
#         {
#             'entity_id': subject_id,
#             'num_chunks': num_chunks,
#             'chunks': sorted_chunks,
#             'chunks_lengths': sorted_lengths
#         }
# """
# important_chunks = get_important_chunks(ds, 50, 100, kas_dataset.get_instance_lengths())

# def get_disjoint_entities(subject_dicts, seed = 406):
#     used_chunks = set()
#     disjoint_entities = []

#     random.seed(seed)

#     shuffled = subject_dicts.copy()
#     random.shuffle(shuffled)

#     for entity in shuffled:
#         entity_chunks = set(entity['chunks'])

#         # Check if entity has any overlapping chunk
#         if used_chunks.isdisjoint(entity_chunks):
#             disjoint_entities.append(entity)
#             used_chunks.update(entity_chunks)

#     return disjoint_entities

# max_seed = 0
# max_len = 0
# for i in range(5000):
#     current_len = len(get_disjoint_entities(important_chunks, i))
#     if current_len > max_len:
#         max_len = current_len
#         max_seed = i

# max_seed, max_len

# disjoint_entities = get_disjoint_entities(important_chunks, max_seed)

# chunks_set = ()

# for entity in disjoint_entities:
#     for chunk in entity['chunks']:
#         if chunk in chunks_set:
#             print("not disjoint")
#             break

all_batches = np.load("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/batch_indices.npy", allow_pickle=True)

# import random

# def sample_injection_points(total_steps, num_points_to_sample, max_num_chunks, interval, seed=None):
#     """
#     Samples unique injection points from a valid starting range to avoid overflow 
#     when assigning chunk indices.

#     Args:
#         total_steps (int): The maximum possible step value (exclusive upper bound).
#         num_points_to_sample (int): Number of injection points to sample.
#         max_num_chunks (int): Maximum num_chunks across all entities.
#         interval (int): Distance between chunk indices.
#         seed (int, optional): Seed for reproducibility.

#     Returns:
#         List[int]: Sorted list of valid injection starting points.
#     """
#     if seed is not None:
#         random.seed(seed)

#     max_valid_start = total_steps - (max_num_chunks - 1) * interval
#     if max_valid_start <= 0:
#         raise ValueError("Interval and chunk size too large for total steps.")

#     if num_points_to_sample > max_valid_start:
#         raise ValueError("Cannot sample more injection points than available valid start points.")

#     sampled_points = random.sample(range(max_valid_start), k=num_points_to_sample)
#     return sorted(sampled_points)


# def assign_indices_to_entities(entities, injection_points, interval):
#     """
#     Assigns indices to each entity starting at a given injection point with spacing.

#     Args:
#         entities (List[dict]): List of entity dicts.
#         injection_points (List[int]): List of sampled injection start points.
#         interval (int): Distance between chunk indices.

#     Returns:
#         Dict[str, List[int]]: Mapping from entity name to list of indices.
#     """
#     if len(entities) != len(injection_points):
#         raise ValueError("Number of entities must match number of injection points.")

#     result = {}

#     for entity, start in zip(entities, injection_points):
#         entity_id = entity['entity_id']
#         num_chunks = entity['num_chunks']
#         indices = [start + i * interval for i in range(num_chunks)]
#         result[entity_id] = indices

#     return result

# interval = 1

# total_number_of_batches = dataloader.total_batches
# injection_points = sample_injection_points(total_number_of_batches, len(disjoint_entities), 100, interval, 0)
# all_injection_points_per_entity = assign_indices_to_entities(disjoint_entities, injection_points, interval)
# # all_injection_points_per_entity

# interval = 1

# total_number_of_batches = dataloader.total_batches
# injection_points = sample_injection_points(total_number_of_batches, len(disjoint_entities), 100, interval, 0)
# all_injection_points_per_entity = assign_indices_to_entities(disjoint_entities, injection_points, interval)
# # all_injection_points_per_entity
# import itertools
# inj = sorted(itertools.chain.from_iterable(all_injection_points_per_entity.values()))[:10]
# print(inj)
inj = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# # for i, batch in enumerate(dataloader):
# #     print(f"******* is  is {i}")
# #     if i == inj[0]:
# #         print(len(batch['index']), batch['index'])
# #         break
# #     if i > 20:
# #         break

# import math

# def shloop(
#     injection_points: List[int],
#     entity_data: dict,
#     batch_to_chunks_map: dict,
#     blacklist = []
# ) -> List[List]:
#     """
#     """
#     # 1. Get entity chunks available for swapping and their lengths
#     fixed_lengths = [2 ** math.ceil(math.log2(length)) for length in entity_data['chunks_lengths']]
    
#     ent_chunk_to_len = dict(zip(entity_data['chunks'], fixed_lengths))
#     ent_len_to_chunk = {v: k for k, v in ent_chunk_to_len.items()}

#     # casting to int but might want to edit this
#     batch_id_to_len = {}
#     batch_len_to_id = {}
#     for batch in injection_points:
#         batch_len = int(32768 / len(batch_to_chunks_map[batch]))
#         batch_id_to_len[batch] = batch_len
#         batch_len_to_id[batch_len] = batch

#     # 2. Calculate the injection span
#     num_chunks = len(entity_data['chunks'])
#     #print(f"Injection span: {list(injection_points)}")
#     if len(injection_points) != num_chunks:
#         f"Entity {entity_data['entity_id']} expected {num_chunks} injection points, but got {len(injection_points)}."
    

#     sb = sorted(batch_len_to_id.keys())   
#     se = sorted(ent_len_to_chunk.keys())
#     #bucket_length = 2 ** math.ceil(math.log2(length))
#     chunks_to_batches = []
#     for len_e in se:
#         for len_b in sb:
#             if len_b == len_e:
#                 #print(len_e, len_b)
#                 chunk_id = ent_len_to_chunk[len_e]
#                 batch_id = batch_len_to_id[len_b]

#                 #print(f"Chunk {chunk_id} with length {len_e} will be swapped with batch {batch_id} with length {len_b}")
#                 # get a random chunk id from the batch
#                 chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
#                 while chunk_id_from_batch in blacklist:
#                     chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])

#                 if [chunk_id, chunk_id_from_batch] in chunks_to_batches or [chunk_id_from_batch, chunk_id] in chunks_to_batches:
#                     print(chunk_id, chunk_id_from_batch, "already in")
                
#                 chunks_to_batches.append([chunk_id, chunk_id_from_batch])
#                 chunks_to_batches.append([chunk_id_from_batch, chunk_id])
#                 #chunks_to_batches[chunk_id] = chunk_id_from_batch # chunk e goes to chunk e' in batch b
#                 #chunks_to_batches[chunk_id_from_batch] = chunk_id # add the symetric mapping

#                 ent_len_to_chunk.pop(len_e) # pop one of the lengths
#                 ent_chunk_to_len.pop(chunk_id) # pop the chunk from the entity and pop one of the lengths
#                 batch_len_to_id.pop(len_b)
#                 batch_id_to_len.pop(batch_id) # pop the batch and the length from the batch
#                 break
                
#     # ranmly match the rest of the chunks
#     for chunk_id, batch_id in zip(ent_chunk_to_len.keys(), batch_id_to_len.keys()):
#         if chunk_id not in chunks_to_batches:
#             chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
#             while chunk_id_from_batch in blacklist:
#                     chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
                    
#             if [chunk_id, chunk_id_from_batch] in chunks_to_batches or [chunk_id_from_batch, chunk_id] in chunks_to_batches:
#                     print(chunk_id, chunk_id_from_batch, "already in")
                
#             chunks_to_batches.append([chunk_id, chunk_id_from_batch])
#             chunks_to_batches.append([chunk_id_from_batch, chunk_id])
            
#             #chunks_to_batches[chunk_id] = chunk_id_from_batch
#             #chunks_to_batches[chunk_id_from_batch] = chunk_id

#     return chunks_to_batches

# full_mapping = []
# blacklist = set()
# for entity in disjoint_entities:
#     blacklist.update(entity['chunks'])

# for i, important_chunk in enumerate(disjoint_entities):
#     pts = all_injection_points_per_entity[important_chunk['entity_id']]

#     # The 'important_chunk' variable is the integer you need.
#     # Pass it directly to your function.
#     res = shloop(
#         pts,
#         important_chunk,
#         all_batches,
#         blacklist
#     )
    
#     # add the chunks already in the injection mapping to a blacklist so they don't get sampled
#     blacklist.update(list(set([r[0] for r in res])))

#     # extend full mapping with the result
#     full_mapping.extend(res)


# normal_dict = {}
# for key, value in full_mapping:
#     normal_dict[key] = value

# total_chunks = sum(entity['num_chunks'] for entity in disjoint_entities)
# print("Combined amount of chunks in disjoint entities:", total_chunks)

# with open('/home/joberant/NLP_2425b/vishnevsky/knowledge-analysis-suite/OLMo-core/swapping_dict.pkl', 'wb') as f:
#     pickle.dump(normal_dict, f)


# To load it back later
with open('/home/joberant/NLP_2425b/vishnevsky/knowledge-analysis-suite/OLMo-core/swapping_dict.pkl', 'rb') as f:
    swapping_dict = pickle.load(f)

tokenizer_config = TokenizerConfig.dolma2()
tokenizer = HFTokenizer(
            tokenizer_config.identifier,
            pad_token_id=tokenizer_config.pad_token_id,
            eos_token_id=tokenizer_config.eos_token_id,
            bos_token_id=tokenizer_config.bos_token_id,
        )

include_instance_metadata = False # Set to true when you want tp retrieve metadata, during training set this to False
# work_dir = "/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"
work_dir = "/home/joberant/NLP_2425b/vishnevsky/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"


dataset_config = NumpyDatasetConfig.glob(
    "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",  # can be globs
    name=NumpyDatasetType.kas_vsl,
    max_sequence_length=2048,
    min_sequence_length=64,
    vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
    tokenizer=tokenizer_config,
    work_dir=str(work_dir),
    include_instance_metadata=include_instance_metadata,
    # swapping_dict = {5265742:2237061, 2237061:5265742},
    swapping_dict = swapping_dict,
)

reordered_dataset = dataset_config.build()

data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=32768,
    seed=0,
    num_workers=8,
    prefetch_factor = 16,
)

dataloader = data_loader_config.build(reordered_dataset)
dataloader.reshuffle(1)

# missing_chunks = []
# for entity in disjoint_entities:
#     for chunk_id in entity['chunks']:
#         if chunk_id not in swapping_dict:
#             missing_chunks.append(chunk_id)

# print(f"Number of missing important chunks: {len(missing_chunks)}")
# if missing_chunks:
#     print("Missing chunk IDs:", missing_chunks[:20])  # show first 20 for brevity
# else:
#     print("All important chunks are present in swapping_dict.")

sorted_keys = list(swapping_dict.keys())
sorted_keys.sort()

from tqdm import tqdm

for i, batch in tqdm(enumerate(dataloader)):
    expected = all_batches[i]
    for j,ch in enumerate(expected):
        if ch in swapping_dict:
            expected[j] = swapping_dict[ch]
    idxs_check = np.array_equal(batch['index'].numpy(), expected)
    if not idxs_check:
        print(f"batch {i} failed")
        print("orig batch: ")
        print(all_batches[i])
        print("new batch: ")
        print(batch['index'])
        break
    # if i==1:
    #     print(batch['input_ids'][123])
    #     print(batch['index'][123])
    #     print(batch['input_ids'].shape)
    #     break
        
