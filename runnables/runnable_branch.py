from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda, RunnableBranch,RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed prompt on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text: {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1 , model , parser)
summary_gen_chain = RunnableSequence(prompt2 , model , parser)

branch_chain = RunnableBranch(
    ( lambda x:len(x.split())>500 , summary_gen_chain),
    RunnablePassthrough(),

)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'ai'})

print(result)
