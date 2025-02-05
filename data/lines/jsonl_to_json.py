# Converts jsonl to json
import json

def jsonl_to_json(jsonl_file_path, json_file_path):
    datas = None
    with open(jsonl_file_path, encoding='utf-8') as f:
        datas = [json.loads(line) for line in f]
    
    with open(json_file_path, 'w') as f:
        f.write(json.dumps(datas, indent=4))

if __name__ == '__main__':
    jsonl_file_path = "./dynamickgqa_test_output.jsonl"
    json_file_path = "./dynamickgqa_test_output.json"
    jsonl_to_json(jsonl_file_path, json_file_path)