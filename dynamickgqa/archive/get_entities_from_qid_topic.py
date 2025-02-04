"""
This file helps get entities (Yago-specific) for other datasets. 
It is also responsible for loading the data, using the entities from qid_topic_entities (Wikidata),
    and getting the Yago-specific entities.
"""

import json
import argparse
from datasets import load_dataset

from utils import run_llm, prepare_dataset, setup_ner, get_entities, get_spacy_entities
from prompt_list import entity_prompt
from yago_func import get_entities_from_labels, get_entities_from_qids

from tqdm import tqdm

def get_entity_qids_from_data(row, entity_col: str):
    """
    Get the entity qids from the data.

    Currently, the format is largely: {Freebase/Wikidata entity: label}
    We want to convert this to {Yago entity: label}
    So, this function will return the entity qids.
    """
    entities = row[entity_col]
    return [qid for qid in entities.keys()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="./cwq.json", help="the path to the data.")
    parser.add_argument("--output_file", type=str,
                        default="cwq_output.jsonl", help="the output file name.")
    args = parser.parse_args()

    topic_entity_col = "qid_topic_entity"

    datas, question_string = prepare_dataset(args.data_path)
    output_data = []

    completed = 0
    for index, row in tqdm(enumerate(datas), total=len(datas)):
        if index < completed: continue

        try:
            # Despite the function being named get_entities, it actually gets the possible entity labels from the results
            qids_to_labels: dict = row[topic_entity_col]
            entity_qids = get_entity_qids_from_data(row, topic_entity_col)

            row["entity_qids"] = entity_qids

            # Get the entities from the labels
            entities = get_entities_from_qids(entity_qids) # Map of qid to entity)

            entity_to_label = {}
            for qid, entity in entities.items():
                if qid in qids_to_labels.keys():
                    entity_to_label[entity] = qids_to_labels[qid]
                else:
                    pass
            # print(entities)
            row["qid_topic_entity"] = entity_to_label

            output_data.append(row)

            if (index+1) % 100 == 0:
                with open(args.output_file, 'a+') as f:
                    for data in output_data:
                        f.write(json.dumps(data) + "\n")
                output_data = []
                print(f"Processed {index+1} rows")
        except Exception as e:
            print(f"Error at index {index}: {e}")
            # Put error message in qid_topic_entity
            row["qid_topic_entity"] = f"Error at index {index}"
            output_data.append(row)
            continue
    
    if len(output_data) > 0:
        with open(args.output_file, 'a+') as f:
            for data in output_data:
                f.write(json.dumps(data) + "\n")
        print(f"Processed {len(datas)} rows")

# Command
# python get_entities_from_qid_topic.py --data_path /home/ubuntu/ClaimBenchKG_Baselines/ToG/data/dump/cwq.json --output_file cwq_output.jsonl
# python get_entities_from_qid_topic.py --data_path /home/ubuntu/ClaimBenchKG_Baselines/ToG/data/cwq.json --output_file cwq_output.jsonl