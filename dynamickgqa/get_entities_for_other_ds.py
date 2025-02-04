"""
This file is used for getting entities for other datasets, similar to the main.py file.
It is also responsible for loading the data, running the LLM, and extracting the entities from the LLM output.
"""
import sys
import json
import argparse
from datasets import load_dataset

from constants import HF_PATH, MAIN_COLUMNS
from utils import run_llm, prepare_dataset, setup_ner, get_entities, get_spacy_entities
from prompt_list import entity_prompt
from yago_func import get_entities_from_labels

from tqdm import tqdm

def load_hf_dataset(data_path = HF_PATH, *, 
                              split: str = 'test', subset: tuple[int, int] = None,
                              columns: list[str] = None):
    dataset = load_dataset(HF_PATH, split=f"{split}[{subset[0]}:{subset[1]}]" if subset else split)
    if columns:
        dataset = dataset.select_columns(columns)
    return dataset

def data_to_json(data_path = HF_PATH, *, json_file_name = "data.json", 
                 split: str = 'test', subset: tuple[int, int] = None, columns: list[str] = None):
    dataset = load_hf_dataset(data_path, split=split, subset=subset, columns=columns)
    bytes_num = dataset.to_json(json_file_name, lines=False, batch_size=dataset.num_rows)
    # bytes_num = 0
    print(f"Data saved to {json_file_name}")
    # print(dataset[0])
    return bytes_num

def load_json(file_path):
    data_files = {"test": file_path}
    re_squad = load_dataset("json", data_files=data_files, split="test")
    print(re_squad[0])


if __name__ == '__main__':
    # First, save the data to a json file, then run the rest of the code
    # data_to_json(HF_PATH, json_file_name="dynamickgqa_test_subset.json", split='test', subset=(0, 10), columns=MAIN_COLUMNS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/ubuntu/ClaimBenchKG_Baselines/ToG/data/cwq.json", help="the path to the data.")
    parser.add_argument("--output_file", type=str,
                        default="/home/ubuntu/ClaimBenchKG_Baselines/ToG/data/cwq.jsonl", help="the output file name.")
    parser.add_argument("--question_string", type=str,
                        default="question", help="the question string.")
    parser.add_argument("--completed", type=int,
                        default=0, help="the number of rows already completed.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature", type=int,
                        default=0, help="the temperature")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    args = parser.parse_args()

    nlp = setup_ner()

    datas, question_string = prepare_dataset(args.data_path, question_string=args.question_string)
    output_data = []

    completed = args.completed
    for index, row in tqdm(enumerate(datas), total=len(datas)):
        if index < completed: continue

        try:
            prompt = entity_prompt + "\n\nQ: " + row[question_string] + "\nA: "
            results = run_llm(prompt, args.temperature, args.max_length, args.opeani_api_keys, args.LLM_type)

            # Despite the function being named get_entities, it actually gets the possible entity labels from the results
            entity_labels = get_entities(results)
            if not entity_labels or len(entity_labels) == 0: entity_labels = get_spacy_entities(nlp, results)

            row["entity_labels"] = entity_labels

            # Get the entities from the labels
            entities = get_entities_from_labels(entity_labels)
            entity_to_label = {entity: label for label, entity in entities.items()}
            # print(entities)
            row["qid_topic_entity"] = entity_to_label

            output_data.append(row)

            if (index+1) % 10 == 0:
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

# Command (Not included)