from langchain_openai import OpenAI
from dotenv import load_dotenv
# no api credits so won't work
load_dotenv()
llm = OpenAI(model='gpt-4')
llm.invoke("Hello how are you?")