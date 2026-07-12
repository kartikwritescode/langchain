from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# llm = HuggingFaceEndpoint(
#     # model='HuggingFaceTB/SmolLM2-360M-Instruct',
#     repo_id='HuggingFaceTB/SmolLM2-360M-Instruct',
#     task='text-generation'
# )

# model = ChatHuggingFace(llm=llm)
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)
# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a summary on {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
# result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':"result1.content"})
# result2 = model.invoke(prompt2)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)