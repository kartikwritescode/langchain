from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt1 = PromptTemplate(
    template='Give me a proper report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize this into 5 points : {report}',
    input_variables=['report']
)
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'gen ai'})

print(result)

print('-----------------------------')

chain.get_graph().print_ascii()