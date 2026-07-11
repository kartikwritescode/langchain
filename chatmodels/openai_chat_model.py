from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# no api credits so won't work
load_dotenv()
llm = ChatOpenAI(model='gpt-4',temperature=1.2,max_completion_tokens=1000)
llm.invoke("Hello how are you?")