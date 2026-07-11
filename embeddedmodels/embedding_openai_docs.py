from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


documents=[
    "Delhi is the capital of India",
    "Kolkata is the captial of West Bengal",
    "Paris is the captial of France"
]
embedding = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

result = embedding.embed_documents(documents)
print(str(result))
