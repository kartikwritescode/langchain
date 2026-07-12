from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()


with open('chains/document.txt','r') as f:
    data = f.readlines()

model1 = ChatGoogleGenerativeAI(model='gemini-3.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

prompt1 = PromptTemplate(
    template='provide me the notes on : \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template=  'provide me 5 ques quiz on : \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template=  'merge the provided notes and quiz into a single document : \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser ,
    'quiz':prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text':data})

print(result)

chain.get_graph().print_ascii()