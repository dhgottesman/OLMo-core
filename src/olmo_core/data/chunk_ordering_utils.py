import random
import math
import pickle

from typing import List

import random

def sample_injection_points(total_steps, num_points_to_sample, max_num_chunks, interval, seed=None, min_start_point=0):
    """
    Samples unique injection points from a valid starting range, respecting a soft interval constraint.

    Args:
        total_steps (int): The maximum possible step value (exclusive upper bound).
        num_points_to_sample (int): Number of injection points to sample.
        max_num_chunks (int): Maximum num_chunks across all entities.
        interval (int): Desired distance between chunk indices (soft constraint).
        seed (int, optional): Seed for reproducibility.
        min_start_point (int): Minimum possible value for a valid injection point.

    Returns:
        List[int]: Sorted list of valid injection starting points.
    """
    if seed is not None:
        random.seed(seed)

    # Ensure min_start_point is within bounds
    if min_start_point < 0 or min_start_point >= total_steps:
        raise ValueError("min_start_point must be within the range [0, total_steps).")

    # Compute maximum possible valid start point
    max_valid_start = total_steps - (max_num_chunks - 1) * interval

    if max_valid_start <= min_start_point:
        # Interval is too large; compute the largest feasible one
        max_feasible_interval = (total_steps - min_start_point) // max(max_num_chunks - 1, 1)
        if max_feasible_interval <= 0:
            raise ValueError("Even the largest possible interval results in overflow. Adjust total_steps or chunk size.")

        print(f"[Warning] Desired interval {interval} too large. Using maximum feasible interval {max_feasible_interval}.")
        interval = max_feasible_interval
        max_valid_start = total_steps - (max_num_chunks - 1) * interval

    valid_range = range(min_start_point, max_valid_start)
    if num_points_to_sample > len(valid_range):
        raise ValueError("Cannot sample more injection points than available valid start points.")

    sampled_points = random.sample(valid_range, k=num_points_to_sample)
    return sorted(sampled_points)



def assign_indices_to_entities(entities, injection_points, interval):
    """
    Assigns indices to each entity starting at a given injection point with spacing.

    Args:
        entities (List[dict]): List of entity dicts.
        injection_points (List[int]): List of sampled injection start points.
        interval (int): Distance between chunk indices.

    Returns:
        Dict[str, List[int]]: Mapping from entity name to list of indices.
    """
    if len(entities) != len(injection_points):
        raise ValueError("Number of entities must match number of injection points.")

    result = {}

    for entity, start in zip(entities, injection_points):
        entity_qid = entity['subject_qid']
        num_chunks = entity['num_chunks']
        indices = [start + i * interval for i in range(num_chunks)]
        result[entity_qid] = indices

    return result


def shloop(
    injection_points: List[int],
    entity_data: dict,
    batch_to_chunks_map: dict,
    blacklist = []
) -> List[List]:
    """
    """
    # 1. Get entity chunks available for swapping and their lengths
    fixed_lengths = [2 ** math.ceil(math.log2(length)) for length in entity_data['chunks_lengths']]
    
    ent_chunk_to_len = dict(zip(entity_data['chunks'], fixed_lengths))
    ent_len_to_chunk = {v: k for k, v in ent_chunk_to_len.items()}

    # casting to int but might want to edit this
    batch_id_to_len = {}
    batch_len_to_id = {}
    for batch in injection_points:
        batch_len = int(32768 / len(batch_to_chunks_map[batch]))
        batch_id_to_len[batch] = batch_len
        batch_len_to_id[batch_len] = batch

    # 2. Calculate the injection span
    num_chunks = len(entity_data['chunks'])
    if len(injection_points) != num_chunks:
        f"Entity {entity_data['entity_id']} expected {num_chunks} injection points, but got {len(injection_points)}."
    

    sb = sorted(batch_len_to_id.keys())   
    se = sorted(ent_len_to_chunk.keys())
    #bucket_length = 2 ** math.ceil(math.log2(length))
    chunks_to_batches = []
    for len_e in se:
        for len_b in sb:
            if len_b == len_e:
                chunk_id = ent_len_to_chunk[len_e]
                batch_id = batch_len_to_id[len_b]

                # get a random chunk id from the batch
                chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
                while chunk_id_from_batch in blacklist:
                    chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])

                # if [chunk_id, chunk_id_from_batch] in chunks_to_batches or [chunk_id_from_batch, chunk_id] in chunks_to_batches:
                #     print(chunk_id, chunk_id_from_batch, "already in")
                
                chunks_to_batches.append([chunk_id, chunk_id_from_batch])
                chunks_to_batches.append([chunk_id_from_batch, chunk_id])
                #chunks_to_batches[chunk_id] = chunk_id_from_batch # chunk e goes to chunk e' in batch b
                #chunks_to_batches[chunk_id_from_batch] = chunk_id # add the symetric mapping

                ent_len_to_chunk.pop(len_e) # pop one of the lengths
                ent_chunk_to_len.pop(chunk_id) # pop the chunk from the entity and pop one of the lengths
                batch_len_to_id.pop(len_b)
                batch_id_to_len.pop(batch_id) # pop the batch and the length from the batch
                break
                
    # ranmly match the rest of the chunks
    for chunk_id, batch_id in zip(ent_chunk_to_len.keys(), batch_id_to_len.keys()):
        if chunk_id not in chunks_to_batches:
            chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
            while chunk_id_from_batch in blacklist:
                    chunk_id_from_batch = random.choice(batch_to_chunks_map[batch_id])
                    
            # if [chunk_id, chunk_id_from_batch] in chunks_to_batches or [chunk_id_from_batch, chunk_id] in chunks_to_batches:
            #         print(chunk_id, chunk_id_from_batch, "already in")
                
            chunks_to_batches.append([chunk_id, chunk_id_from_batch])
            chunks_to_batches.append([chunk_id_from_batch, chunk_id])
            
            #chunks_to_batches[chunk_id] = chunk_id_from_batch
            #chunks_to_batches[chunk_id_from_batch] = chunk_id

    return chunks_to_batches

def create_swapping_dict_for_steps_interval(interval: int, entities, batch_to_chunks_map: dict, total_number_of_batches: int = 109672, max_chunks_per_entity: int = 100, seed: int = 0, save_file_path: str | None = None) -> dict: 
        
    injection_points = sample_injection_points(total_number_of_batches, len(entities), max_chunks_per_entity, interval, seed)
    all_injection_points_per_entity = assign_indices_to_entities(entities, injection_points, interval)

    full_mapping = []
    blacklist = set()

    for entity in entities:
        blacklist.update(entity['chunks'])

    for i, important_chunk in enumerate(entities):
        pts = all_injection_points_per_entity[important_chunk['subject_qid']]

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

    if save_file_path:
        with open(save_file_path + f'swapping_dict_interval_{interval}.pkl', 'wb') as f1:
            pickle.dump(swapping_dict, f1)
        with open(save_file_path + f'injected_batched_per_entity_interval_{interval}.pkl', 'wb') as f2:
            pickle.dump(all_injection_points_per_entity, f2)
    
    return swapping_dict

def create_swapping_dict_for_injection_points(interval: int, entities, all_injection_points_per_entity: dict, batch_to_chunks_map: dict, save_file_path: str | None = None) -> dict: 

    full_mapping = []
    blacklist = set()

    for entity in entities:
        blacklist.update(entity['chunks'])

    for i, important_chunk in enumerate(entities):
        pts = all_injection_points_per_entity[important_chunk['subject_qid']]

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

    if save_file_path:
        with open(save_file_path + f'swapping_dict_interval_{interval}.pkl', 'wb') as f1:
            pickle.dump(swapping_dict, f1)
        with open(save_file_path + f'injected_batched_per_entity_interval_{interval}.pkl', 'wb') as f2:
            pickle.dump(all_injection_points_per_entity, f2)
    
    return swapping_dict


def get_important_chunks(dataset, min_num_chunks, max_num_chunks, instance_lengths):
    # Filter the dataset
    filtered_dataset = dataset['train'].filter(
        lambda example: min_num_chunks <= example['subject_num_chunks'] <= max_num_chunks
    )   

    # Create list of dictionaries with subject info and chunk lengths
    result_list = []    

    for example in filtered_dataset:
        
        subject_name = example['subj']
        subject_qid = example['s_uri']
        chunks = example['subject_chunks']
        num_chunks = example['subject_num_chunks']

        chunk_lengths = instance_lengths[chunks]

        # Sort chunks by their lengths (descending order)
        if len(chunk_lengths) > 0:
            # Create pairs of (chunk, length) and sort by length
            chunk_length_pairs = list(zip(chunks, chunk_lengths))
            chunk_length_pairs.sort(key=lambda x: x[1], reverse=True)

            # Separate back into sorted chunks and lengths
            sorted_chunks = [pair[0] for pair in chunk_length_pairs]
            sorted_lengths = [pair[1] for pair in chunk_length_pairs]
        else:
            sorted_chunks = chunks
            sorted_lengths = chunk_lengths

        subject_dict = {
            'subject_qid': subject_qid,
            'entity_name': subject_name,
            'num_chunks': num_chunks,
            'chunks': sorted_chunks,
            'chunks_lengths': sorted_lengths
        }
        
        result_list.append(subject_dict)    

    # Sort the list by number of chunks (descending order)
    result_list.sort(key=lambda x: x['num_chunks'], reverse=True)

    return result_list

def get_disjoint_entities(subject_dicts, seed = 406):
    used_chunks = set()
    disjoint_entities = []

    random.seed(seed)

    shuffled = subject_dicts.copy()
    random.shuffle(shuffled)

    for entity in shuffled:
        entity_chunks = set(entity['chunks'])

        # Check if entity has any overlapping chunk
        if used_chunks.isdisjoint(entity_chunks):
            disjoint_entities.append(entity)
            used_chunks.update(entity_chunks)

    return disjoint_entities
