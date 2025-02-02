from datasets import load_dataset

from constants import HF_PATH, MAIN_COLUMNS

def load_hf_dataset(data_path = HF_PATH, *, 
                              split: str = 'test', subset: tuple[int, int] = None,
                              columns: list[str] = None):
    dataset = load_dataset(HF_PATH, split=f"{split}[{subset[0]}:{subset[1]}]" if subset else split)
    if columns:
        dataset = dataset.select_columns(columns)
    return dataset

def data_to_json(data_path = HF_PATH, *, json_file_name = "data.json", split, subset, columns):
    dataset = load_hf_dataset(data_path, split=split, subset=subset, columns=columns)
    bytes_num = dataset.to_json(json_file_name, lines=False)
    print(f"Data saved to {json_file_name}")
    # print(dataset[0])
    return bytes_num

def load_json(file_path):
    data_files = {"test": file_path}
    re_squad = load_dataset("json", data_files=data_files, split="test")
    print(re_squad[0])

if __name__ == '__main__':
    data = data_to_json(split='test', subset=(0, 10), columns=MAIN_COLUMNS)
    print(data)