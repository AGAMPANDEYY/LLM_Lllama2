from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader #you can also use unstructured datafiles
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
DATA_PATH="channelhydraulics.pdf"
DATA_FAISS_PATH="vectorstores/db_faiss"
def create_vector_db():
  loader=DirectoryLoader(DATA_PATH, glob="*.pdf",loader_cls=PyPDFLoader)
  documents=loader.load()
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
  texts=text_splitter.split_documents(documents)

  embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})#model_kwargs

  db=FAISS.from_documents(texts,embedding)
  db.save_local(DATA_FAISS_PATH)

  if __name__== '__main__':
    create_vector_db()