from langchain_community.document_loaders import TextLoader

loader = TextLoader('gemini_models.txt')

docs =loader.load()
# print(docs)
print(docs[0].page_content)