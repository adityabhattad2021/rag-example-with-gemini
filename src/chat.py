from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
# import re


load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_pdf_text(pdf_file="src/assets/sample_doc_3.pdf", number_of_pages=50, debug=False):
  if not os.path.exists(pdf_file):
    raise FileNotFoundError(
        f"The specified PDF file does not exist: {pdf_file}")
  text = ""
  pdf_reader = PdfReader(pdf_file)
  if number_of_pages > len(pdf_reader.pages):
    number_of_pages = len(pdf_reader.pages)
  for page in pdf_reader.pages[:number_of_pages]:
    page_text = page.extract_text()
    if isinstance(page_text, str):
      text += page_text
    else:
      text += str(page_text)
  # text = text.strip()
  # text = re.sub(r'\s+', '', text)
  if debug:
    print(text)
  return text


# def get_text_chunks(text, chunk_size=10,chunk_overlap=1,debug=False):
#     text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#         chunk_size=100,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     chunks = text_splitter.split_text("The aim of this project is to produce age-appropriate non-fiction books for children from birth to age 12. These books are richly illustrated with photographs, diagrams, sketches, and original drawings. Wikijunior books are produced by a worldwide community of writers, teachers, students, and young people all working together. The books present factual information that is verifiable. You are invited to join in and write, edit, and rewrite each book and chapter to improve its content. Our books are distributed free of charge under the terms of the Creative Commons Attribution-ShareAlike License.")
#     return chunks

def manual_split_text(text, chunk_size=100, chunk_overlap=10, debug=False):
  # Calculate the start index for each chunk
  start_index = 0
  end_index = chunk_size
  chunks = []

  while start_index < len(text):
    print(start_index, end_index, len(text))
    end_index = min(len(text), start_index + chunk_size)
    chunk = text[start_index:end_index]
    if debug:
      print(chunk)
    chunks.append(chunk)
    if end_index >= len(text):
      break
    start_index = end_index - chunk_overlap

  return chunks


def get_vector_store(text_chunks, debug=False):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  vector_store = FAISS.from_texts(text_chunks, embeddings)
  if debug:
    print(vector_store)
  vector_store.save_local("faiss_index")


def get_qa_chain(debug=False):
  prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    """
  model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
  prompt = PromptTemplate(template=prompt_template,
                          input_variables=["context", "question"])
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  if debug:
    print(prompt, model, chain)
  return chain


def generate_reply(question):
  embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
  db = FAISS.load_local("faiss_index", embeddings)
  docs = db.similarity_search(question)
  chain = get_qa_chain()
  response = chain(
      {"input_documents": docs, "question": question},
      return_only_outputs=True
  )
  return response["output_text"]


def main():
  text = get_pdf_text(debug=False)
  print(text)
  # chunks=manual_split_text(text,chunk_size=10000,chunk_overlap=1000,debug=True)
  # get_vector_store(chunks,debug=True)
  # generate_reply("What is health regarded as?")


if __name__ == "__main__":
  main()
