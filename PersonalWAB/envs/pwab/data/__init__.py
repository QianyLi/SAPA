import json
import os
import glob

FOLDER_PATH = os.path.dirname(__file__)


def _require_file(name):
    path = os.path.join(FOLDER_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PersonalWAB data file is missing: {path}. "
            "Download the benchmark data from "
            "https://hongrucai.github.io/PersonalWAB/download and place it "
            "under PersonalWAB/envs/pwab/data/."
        )
    return path


def merge_json_files(file_pattern):
    merged_data = {}

    for json_file in glob.glob(file_pattern):
        with open(json_file, 'r') as f:
            data = json.load(f)
            merged_data.update(data)  
        #print(f"Loaded {json_file}")
    
    return merged_data

_require_file("user_instructions.json")
_require_file("user_profiles.json")
if not glob.glob(os.path.join(FOLDER_PATH, "all_products_part_*.json")):
    raise FileNotFoundError(
        "PersonalWAB product data is missing. Download the benchmark data and "
        "place all_products_part_*.json under PersonalWAB/envs/pwab/data/."
    )
if not glob.glob(os.path.join(FOLDER_PATH, "user_history_part_*.json")):
    raise FileNotFoundError(
        "PersonalWAB history data is missing. Download the benchmark data and "
        "place user_history_part_*.json under PersonalWAB/envs/pwab/data/."
    )

all_products = merge_json_files(os.path.join(FOLDER_PATH, "all_products_part_*.json"))
user_history = merge_json_files(os.path.join(FOLDER_PATH, "user_history_part_*.json"))

data = {
    "tasks": json.load(open(_require_file("user_instructions.json"))),
    "user_profile": json.load(open(_require_file("user_profiles.json"))),
    "user_history": user_history,
    "all_products": all_products
}
