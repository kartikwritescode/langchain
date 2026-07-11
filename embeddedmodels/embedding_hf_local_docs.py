from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={"device": "cuda"})

documents=[
    "Delhi is the capital of India",
    "Kolkata is the captial of West Bengal",
    "Paris is the captial of France"
]
vector = embeddings.embed_documents(documents)
print(str(vector))