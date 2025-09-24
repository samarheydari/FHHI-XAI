import json
import requests
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get entity from Orion")
    parser.add_argument("--entity_type", type=str, help="Entity type to get from Orion")

    args = parser.parse_args()
    entity_type = args.entity_type

    orion_url = "https://orion.tema.digital-enabler.eng.it"

    get_entity_url = f"{orion_url}/ngsi-ld/v1/entities?type={entity_type}"

    # need application/json header
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(get_entity_url, headers=headers)

    json_response = response.json()


    print(json.dumps(json_response, indent=2))