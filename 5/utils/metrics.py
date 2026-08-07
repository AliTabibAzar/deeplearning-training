import json

def load_results():
    try:
        with open('results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None