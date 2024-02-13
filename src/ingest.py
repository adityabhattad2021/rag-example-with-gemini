from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.globals import set_debug
from util import get_absolute_path
import os


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
set_debug(True)




def generate_vector_store():
  loader = DirectoryLoader(get_absolute_path("./assets"),glob="*.pdf",loader_cls=PyPDFLoader)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
  texts = text_splitter.split_documents(documents)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  vector_store = FAISS.from_documents(texts,embeddings)
  vector_store.save_local(get_absolute_path("vectorstore/db_faiss"))


if __name__ == "__main__":
  generate_vector_store()
