from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings


sample = '''
Artificial Intelligence (AI) has rapidly evolved from being a research topic to becoming a part of everyday life. AI systems now assist people in communication, healthcare, education, transportation, finance, and entertainment. The growth of machine learning and deep learning has enabled computers to recognize images, understand human language, and even generate creative content.

Machine Learning is a subset of Artificial Intelligence that focuses on enabling computers to learn patterns from data instead of relying on explicitly programmed rules. Supervised learning, unsupervised learning, and reinforcement learning are the three major categories of machine learning. Supervised learning uses labeled datasets, while unsupervised learning attempts to identify hidden patterns without labels. Reinforcement learning trains an agent by rewarding desirable actions and penalizing undesirable ones.

Deep Learning is a specialized branch of machine learning that uses neural networks with multiple layers. These networks can automatically extract meaningful features from raw data, making them extremely useful for tasks such as image classification, object detection, speech recognition, and natural language processing. Popular deep learning frameworks include TensorFlow and PyTorch.

Natural Language Processing (NLP) focuses on enabling computers to understand and generate human language. NLP applications include machine translation, sentiment analysis, text summarization, chatbots, and question-answering systems. Recent advancements in transformer architectures have significantly improved the performance of NLP models.

Large Language Models (LLMs) such as GPT, Gemini, Claude, and Llama are built using transformer architectures. These models are trained on enormous datasets containing books, websites, articles, and code repositories. During training, the models learn statistical relationships between words, allowing them to predict the next token in a sequence. After training, instruction tuning and reinforcement learning from human feedback further improve their conversational abilities.

Retrieval-Augmented Generation (RAG) combines language models with external knowledge sources. Instead of relying solely on the information stored during training, a RAG system retrieves relevant documents from a vector database and provides them as context to the language model. This approach improves factual accuracy and allows the model to answer questions about private or recently updated data.

A typical RAG pipeline begins by loading documents using document loaders. These documents are then divided into smaller chunks using text splitters such as RecursiveCharacterTextSplitter. Each chunk is converted into a numerical representation called an embedding. Embeddings capture the semantic meaning of text and allow similar pieces of information to be located efficiently.

Once embeddings are created, they are stored inside a vector database such as Chroma, Pinecone, FAISS, Weaviate, or Milvus. During retrieval, the user's query is converted into an embedding, and similarity search is performed to identify the most relevant document chunks. These retrieved chunks are passed to the language model as additional context before generating a response.

LangChain is a popular framework for building LLM-powered applications. It provides components for document loading, text splitting, embeddings, vector stores, retrievers, prompt templates, chains, memory, agents, and tools. Developers can combine these components to create sophisticated AI applications with relatively little code.

Prompt engineering plays an important role in obtaining reliable outputs from language models. A good prompt clearly specifies the task, provides necessary context, defines constraints, and may even include examples. Few-shot prompting demonstrates the desired input-output behavior through examples, while zero-shot prompting relies entirely on the model's pre-trained knowledge.

Embeddings transform text into high-dimensional vectors. Unlike keyword-based search, vector similarity search identifies documents based on meaning rather than exact word matches. This allows a search for "automobile" to retrieve documents containing the word "car," even if the exact keyword is absent.

Vector databases are optimized for storing and searching embeddings efficiently. They use approximate nearest neighbor algorithms such as HNSW or IVF to retrieve the closest vectors quickly, even when millions of vectors are stored.

Modern AI applications frequently integrate external APIs. Weather APIs, financial APIs, mapping services, and search engines can all provide real-time information that complements the language model's existing knowledge. Agent-based systems can automatically decide which tools to invoke depending on the user's request.

Responsible AI development requires careful consideration of privacy, fairness, transparency, and security. Models should avoid generating harmful content, leaking confidential information, or reinforcing unfair biases. Developers often implement guardrails, moderation systems, and human review processes to improve reliability.

Cloud computing platforms such as AWS, Google Cloud Platform, and Microsoft Azure provide scalable infrastructure for AI applications. GPU instances accelerate model training and inference, while managed database services simplify deployment. Containerization with Docker and orchestration using Kubernetes help ensure applications remain portable and scalable.

The future of artificial intelligence is expected to include more capable multimodal systems that understand text, images, audio, and video simultaneously. Advances in reasoning, planning, robotics, and scientific discovery may further expand the impact of AI across industries. However, continued progress will require balancing innovation with ethical responsibility and ensuring that AI technologies remain beneficial to society.
'''


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

docs = text_splitter.create_documents([sample])

print(len(docs))

for doc in docs:
    print("=" * 50)
    print(doc.page_content)