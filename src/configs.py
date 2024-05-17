import json

with open("secrets/gen-ai-hub-service-key.json", "r") as f:
    gen_ai_hub_service_key = json.load(f)

with open("secrets/hana-secrets.json", "r") as f:
    hana_secrets = json.load(f)
