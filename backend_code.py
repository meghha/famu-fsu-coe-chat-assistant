# Generic stuff
import os
from IPython.display import Markdown
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from operator import itemgetter

# Gemini libraries (llm and embeddings)
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

# Document loader
from langchain_community.document_loaders import PyMuPDFLoader

# Document splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store - Chroma
from langchain_community.vectorstores import Chroma

# Compression Retrievers
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Memory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document,Document

# Prompt Templates
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import get_buffer_string

# Output parsers
from langchain_core.output_parsers import StrOutputParser

# constants
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
VECTOR_STORE_LOCATION = "./chroma_db"

# Load the .env file
load_dotenv()

# Document loader
def doc_loader(website_link):
    loader = PyMuPDFLoader(website_link)
    pages = loader.load()
    return pages

# Text splitting
def doc_splitter(chunk_size, chunk_overlap,pages):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],    
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=pages)
    return chunks

def google_embeddings():
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
    return google_embeddings

# Chroma vector store and save to disk
def vector_store(chunks):
    vector_db = Chroma.from_documents(
                embedding=google_embeddings(),
                documents = chunks,
                persist_directory="./chroma_db"
            )
    return vector_db

def vector_db_creation():
    # load from disk if path exists
    if os.path.exists(VECTOR_STORE_LOCATION):
        vector_db = Chroma(persist_directory = "./chroma_db",embedding_function=google_embeddings())
    else:
        pages = doc_loader("https://eng.famu.fsu.edu/sites/g/files/upcbnu1751/files/pdfs/FAMU-FSU-Engineering-Student-Handbook-2022-2023.pdf")
        chunks = doc_splitter(10000,1000,pages)
        vector_db = vector_store(chunks)
    return vector_db

# Create the retriever along with a contextual retriever using cohere that ranks the results from the retriever from most relevant to least relevant and removes the results that are not contextually relevant
def cohere_contextual_compression_retriever(vdb, k,cohere_api_key,top_n):
    base_retriever = vdb.as_retriever(search_type = "mmr", k = k)
    compressor = CohereRerank(model="rerank-english-v3.0", cohere_api_key=cohere_api_key, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    return compression_retriever

def gemini_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.4)
    return llm

def _combine_documents(docs, document_prompt, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

def chatbot(query,vdb,memory):
    llm = gemini_llm()
    cohere_retriever = cohere_contextual_compression_retriever(vdb=vdb, k=4, cohere_api_key= os.getenv("COHERE"), top_n=8)
                                                
    # memory
    # memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)

    # runnable pass through is a helper function that gets assigned the memory to load all the memory and get the chat history, so that it can be used a part of a chain
    loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))

    # Template for rephrasing follow-up questions
    chat_history_prompt = PromptTemplate(input_variables=['chat_history', 'question'], 
                                          template = """You are a chat assistant for the College of Engineering jointly managed by Florida A&M University and Florida State University. Given the chat history provided below, and the follow up question, rephrase the folllow up question to be a standalone question. \n\n
                                          Chat History:\n{chat_history}\n
                                          Follow Up question: {question}\n
                                          Standalone question:"""        
                                          )
    
    template = f"""Use the context provided to answer the question at the end. The context will include the chat history and relevant information.

<context>
You are a chat assistant designed to help prospective undergraduate applicants, current students, and incoming undergraduate students by answering their queries about the FAMU-FSU College of Engineering regarding courses, policies, prerequisites, and other related details. Please use the information provided from the undergraduate handbook, including the chat history, to respond accurately.
{{chat_history}}

{{context}}
</context>

Question: {{question}}.
"""

     
    answer_prompt_variables = {
            "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
        }
        
    answer_prompt = ChatPromptTemplate.from_template(template)
    # Chain for creating standalone questions
    standalone_question_chain = {"standalone_question": {"question": lambda x: x["question"],
                                                         "chat_history": lambda x: get_buffer_string(x["chat_history"])} 
                                                         | chat_history_prompt | llm | StrOutputParser()}
    
    # Retrieve documents based on standalone question
    retrieved_documents = {"docs": itemgetter("standalone_question") | cohere_retriever
                           , "question": lambda x: x["standalone_question"]}
    


    # Chain to generate the answer
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata 
        "docs": lambda x: [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # Combining chains for the final process
    chain_question = loaded_memory | standalone_question_chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    # Invoke the chain with the query and get the answer
    response = conversational_retriever_chain.invoke({"question":query})
    
    answer = response['answer'].content
    memory.save_context( {"question": query}, {"answer": answer} )
    return answer, memory
