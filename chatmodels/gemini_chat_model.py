from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# no api credits so won't work
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash',temperature=1.2)
result = model.invoke("Can you write a cool joke?")

print(result.content[0]["text"])

