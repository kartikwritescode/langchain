# feedback -> analyze -> condition positive -> feedback
#                                  negative -> email or anything , here another response


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import Literal
from langchain_core.runnables import RunnableBranch , RunnableLambda  

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback )

prompt1= PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive and negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)
prompt2= PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback'],
)
prompt3= PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback'],
)


classifier_chain = prompt1 | model | parser2

# branch_chain = RunnableBranch(
#     (condition1 , chain),
#     (condition1 , chain),
#     default chain
# )


branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive' , prompt2 | model | parser ),
    (lambda x:x.sentiment=='negative' , prompt3 | model | parser ),
    RunnableLambda(lambda x:"couldn't find sentiment")
)

chain = classifier_chain | branch_chain


result = chain.invoke({'feedback':'This is the worst smartphone'})

print(result)

chain.get_graph().print_ascii()


# In this code the original feedback is not being sent to the branch and hence the context couldn't be captured -> to do so add another variable -> left coz am lazy