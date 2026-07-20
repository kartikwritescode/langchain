# this code violates the free tier RPM limit - req per min (5) and thus fails since it fires both req simultaneously
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableParallel

load_dotenv()


prompt1 = PromptTemplate(
    template='Write a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write a linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1 , model , parser),
    'post':RunnableSequence(prompt2 , model , parser),
})

result = parallel_chain.invoke({'topic':'AI'})

print(result)c