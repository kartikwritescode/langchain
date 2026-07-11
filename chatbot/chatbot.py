from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

chat_history=[
    SystemMessage(content="You are a helpful AI assistant ")
]
messages=[
    SystemMessage(content="")
]
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

while True:
    user_input = input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input =='exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content[0]["text"]))
    print("AI:",result.content[0]["text"])
print(chat_history)