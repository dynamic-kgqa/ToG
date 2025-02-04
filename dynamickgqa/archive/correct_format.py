"""
This file is used to correct the format of qid_topic_entity in the json files for dynamickgqa. 
"""
import json

DATASET_PATH = './dynamickgqa_test_output.jsonl'
OUTPUT_PATH = './dynamickgqa_test_output_corrected.jsonl'

datas = None
with open(DATASET_PATH, encoding='utf-8') as f:
    datas = [json.loads(line) for line in f]

THRESHOLD = 1000
output_data = []
for index, row in enumerate(datas):
    if 'qid_topic_entity' in row:
        qid_topic_entity = row['qid_topic_entity']
        fixed_qid_topic_entity = {}
        for label, entity in qid_topic_entity.items():
            fixed_qid_topic_entity[entity] = label
        row['qid_topic_entity'] = fixed_qid_topic_entity
    
    output_data.append(row)
    if (index+1) % THRESHOLD == 0:
        with open(OUTPUT_PATH, 'a+') as f:
            for data in output_data:
                f.write(json.dumps(data) + '\n')
        output_data = []
