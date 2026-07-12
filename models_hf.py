from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient()

models = client.list_deployed_models()

print(models)