from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
# no api credits so won't work
load_dotenv()
llm = ChatAnthropic(model='claude-3-5-sonnet-latest',temperature=1.2)
msg = llm.invoke("Hello how are you?")
print(msg.content)