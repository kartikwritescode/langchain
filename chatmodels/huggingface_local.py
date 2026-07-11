from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    max_new_tokens=256,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=pipe)

result = llm.invoke("What is LangChain?")

print(result)