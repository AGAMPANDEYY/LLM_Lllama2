from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import conversational_retrieval #for chat history, used this, we can use other chains
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader #you can also use unstructured datafiles
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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

custom_prompt_template=""" Use the following peieces of information to answer user's question.If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for conversation retreival for each vector stores.
    """
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])

    return prompt

#loading the llm model

def retrieval_conversation(llm,prompt,db):
    conversation_chain=conversational_retrieval.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        #since we only want llama2 to return output from our pdf so return source documents
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )
    return conversation_chain

def loadllm():
    llm=CTransformers(
        model=r"C:\Users\dell\Downloads\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1096,
        temperature=0.5
    )
    return llm
def qa_bot():
  #loading the embeddings stored in FAISS database and calling the model with the custom_prompt set above and passing the model with others through a retrival conversationa chain of langchain
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})  # model_kwargs
    db=FAISS.load_local(DATA_FAISS_PATH,embedding)
    llm=loadllm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_conversation(llm,qa_prompt,db)

    return qa

def final_result(query):
    qa_Result=qa_bot()
    response=qa_Result({'query':query})
    return response

#chainlit part implemetation

@cl.on_chat_start
async  def start():
    chain=qa_bot()
    msg=cl.Message(content="Starting the conversation...")
    await msg.send()
    msg.content="Hi, welcome to Conversation"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    res=await chain.acall(message,callbacks=[cb])
    answer=res["result"]
    sources=res["source_documents"]

    if sources:
        answer+=f"\nSources: "+ str(sources)
    else:
        answer+=f"\nNo sources found "
    await cl.Message(content=answer).send()
