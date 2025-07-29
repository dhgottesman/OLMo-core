import numpy as np
import os
from datasets import load_dataset
import json
from tqdm import tqdm

checkpoints_path = "/home/morg/students/gottesman3/knowledge-analysis-suite/olmes/hp_final/kas_1B_1_checkpoints"



cache_base = "/home/joberant/NLP_2425b/shirab6"

# Set the cache directory to your actual home
os.environ["HF_HOME"] = cache_base
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")


with open("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/batch_indices.npy","rb") as f:
    batch_indices = np.load(f, allow_pickle=True)

ds = load_dataset("dhgottesman/popqa-kas", split="train")
popqa_questions = ds.to_pandas()

# subject_chunks --> shared_chunks
popqa_questions = popqa_questions[["s_uri", "shared_chunks"]].drop_duplicates("s_uri")

from collections import defaultdict

chunk_to_uris = defaultdict(list)
for _, row in popqa_questions.iterrows():
    for chunk_id in row["shared_chunks"]:
        chunk_to_uris[chunk_id].append(row["s_uri"])


def load_merged_popqa(filepath):
    popqa_performance = {}
    with open(filepath, 'r') as f:
        for line in f:
            obj = json.loads(line)
            doc = obj.get("doc", {})
            s_uri = doc.get("s_uri")
            exact_match = obj.get("exact_match", 0.0)
            correct = int(exact_match) if not exact_match is None else 0
            if not s_uri in popqa_performance:
                popqa_performance[s_uri] = {"questions": 1, "correct": correct}
            else:
                popqa_performance[s_uri]["questions"] += 1
                popqa_performance[s_uri]["correct"] += correct
    return popqa_performance

def merge_checkpoints_dicts(ck1: dict[str,int], ck2: dict[str,int]) -> dict:
    merged = {}

    # Get all unique keys from both dicts
    all_keys = set(ck1.keys()).union(ck2.keys())

    for key in all_keys:
        if key in ck1 and key in ck2:
            merged[key] = {
                "occurences": ck1[key]["occurences"] + ck2[key]["occurences"],
                "last_occurence": max(ck1[key]["last_occurence"], ck2[key]["last_occurence"]),
            }
        elif key in ck1:
            merged[key] = ck1[key]
        else:
            merged[key] = ck2[key]

    return merged

def merge_occurence_with_performance(occurence: dict, performance: dict):
    for key in performance.keys():
        performance[key] = performance[key] | occurence.get(key,{})
    return performance



def entities_in_chunk(chunk_id):
    return chunk_to_uris.get(chunk_id, [])

from concurrent.futures import ThreadPoolExecutor, as_completed

def checkpoint_analysis_parallel(start_batch: int, end_batch: int, entities_data, max_workers: int = 8):
    checkpoint_batches_indices = batch_indices[start_batch:end_batch]

    def process_batch(batch_index, chunk_ids):
        local_entities = {}
        for chunk in chunk_ids:
            for entity in entities_in_chunk(chunk):
                if entity not in local_entities:
                    local_entities[entity] = {"occurences": 1, "last_occurence": batch_index + start_batch}
                else:
                    local_entities[entity]["occurences"] += 1
                    local_entities[entity]["last_occurence"] = batch_index + start_batch
        return local_entities

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_batch, i, batch)
            for i, batch in enumerate(checkpoint_batches_indices)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            batch_result = future.result()
            for entity, stats in batch_result.items():
                if entity not in entities_data:
                    entities_data[entity] = stats
                elif "occurences" not in entities_data[entity]:
                    entities_data[entity]["occurences"] = 0
                    entities_data[entity]["last_occurence"] = 0
                else:
                    entities_data[entity]["occurences"] += stats["occurences"]
                    entities_data[entity]["last_occurence"] = max(
                        entities_data[entity]["last_occurence"],
                        stats["last_occurence"]
                    )

    return entities_data
    
entities_data = {}
# for checkpoint in tqdm(range(1,11)):
#     start_batch = (checkpoint - 1) * 10000
#     end_batch = checkpoint * 10000
#     output_path = os.path.join("/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/outputs",f"checkpoint_{checkpoint}.json")
#     checkpoint_full_path = os.path.join(checkpoints_path,f"step{end_batch}","task-000-popqa_cloze-merged-results.jsonl")
#     performance = load_merged_popqa(checkpoint_full_path)
#     entities_data = checkpoint_analysis_parallel(start_batch,end_batch,entities_data)
#     all_data = merge_occurence_with_performance(entities_data, performance)
#     with open(output_path,"w") as f:
#         json.dump(all_data, f)

with open("/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/outputs/checkpoint_10.json","r") as f:
    entities_data = json.load(f)
output_path = os.path.join("/home/joberant/NLP_2425b/shirab6/knowledge-analysis-suite/OLMo-core/outputs",f"checkpoint_final.json")
checkpoint_full_path = os.path.join("/home/morg/students/gottesman3/knowledge-analysis-suite/olmes/hp_final/kas_1B_1","task-000-popqa_cloze-merged-results.jsonl")
performance = load_merged_popqa(checkpoint_full_path)
entities_data = checkpoint_analysis_parallel(100000,batch_indices.shape[0],entities_data)
all_data = merge_occurence_with_performance(entities_data, performance)
with open(output_path,"w") as f:
    json.dump(all_data, f)












