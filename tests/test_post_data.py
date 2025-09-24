import argparse
import json
import requests

from entity_paths import entity_type_to_entity_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_type", type=str, help="Entity type to test", choices=entity_type_to_entity_path.keys())
    parser.add_argument("--cloud", action="store_true", help="Use cloud endpoint")

    args = parser.parse_args()

    entity_type = args.entity_type
     
    entity_path = entity_type_to_entity_path[entity_type]

    with open(f"{entity_path}") as f:
        entity_data = json.load(f)

    if args.cloud:
        url = "https://tema-project.ddns.net/tfa02/post_data"
    else:
        hostname = "http://localhost"
        port = 8080
        endpoint = "tfa02/post_data"
        url = f"{hostname}:{port}/{endpoint}"

    response = requests.post(url, json=entity_data)

    try:
        print(response.json())
    except:
        print(f"Failed to get response as a json")
        print(response.text)
