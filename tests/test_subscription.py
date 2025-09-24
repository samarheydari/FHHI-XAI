import argparse
import json
import requests

from entity_paths import entity_type_to_entity_path

if __name__ == "__main__":

    # Create the subscription to the test entity
    # entity_type = "PersonVehicleDetectionTest"


    # Send the entity to the Orion Context Broker
    entity_path = "PersonVehicleDetectionTest.json"

    with open(entity_path, "r") as f:
        entity_data = json.load(f)

    # Send the test  
    update_entity_url = "https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entityOperations/upsert"
    response = requests.post(update_entity_url, json=entity_data)

    print(f"Response status code: {response.status_code}")
    print(response.text)
