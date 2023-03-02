import json


def load_config(path: str) -> dict:
    with open(path, 'r') as fp:
        return json.load(fp)
