from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../gemini_models.txt')

docs = loader.lazy_load()

splitter = CharacterTextSplitter(
    chunk_overlap=0,
    separator='',
    chunk_size=1000,    
)

result = splitter.split_documents(docs)

print(result[0].page_content)


  
