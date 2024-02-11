from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.globals import set_debug

set_debug(True)

import os



load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


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
    print(vector_store)
    vector_store.save_local("../faiss_index_new")




def generate_reply(question,debug=False):
  embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
  db = FAISS.load_local("/workspaces/rag-example-with-gemini/faiss_index", embeddings)
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
  print(retriever.invoke("What is color of sunsets on mars?"))
  # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)
  # chain = RetrievalQA.from_chain_type(
  #    llm=llm,
  #    retriever=retriever,
  #    chain_type="stuff"
  # )
  # chain.run("What is on 20 feb")
  
    # chain = get_qa_chain()
    # response = chain(
    #     {"input_documents": docs, "question": question},
    #     return_only_outputs=True
    # )
    # return response["output_text"]


def main():
  
  generate_vector_store("/workspaces/rag-example-with-gemini/src/sample.txt",chunk_size=100)
  # generate_reply("When is world health day?")


if __name__ == "__main__":
  main()
