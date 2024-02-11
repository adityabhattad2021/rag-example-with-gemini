from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.globals import set_debug
import os


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
set_debug(True)

def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, relative_path)
    return os.path.abspath(absolute_path)



def generate_vector_store(file_path, chunk_size=100,chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs,embeddings)
    vector_store.save_local("faiss_index_new")



def generate_reply(question):
  embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
  db = FAISS.load_local(get_absolute_path('../faiss_index_new'), embeddings)
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
  llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)
  chain = RetrievalQA.from_chain_type(
     llm=llm,
     retriever=retriever,
     chain_type="stuff"
  )
  result=chain.run(question)
  return result


def main():
   pass

if __name__ == "__main__":
  main()
