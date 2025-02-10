from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

app=Flask(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['PINECONE_API_KEY']=os.getenv('PINECONE_API_KEY')
llm = GoogleGenerativeAI(model="gemini-pro",temperature=0.1)

embeddings=download_hugging_face_embeddings()

index_name = "medibot"
docsearch=Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriver=docsearch.as_retriever(search_typr="similarity",search_kwargs={"k":3})


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answering_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriver,question_answering_chain)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form['msg']
    input=print(msg)
    response=rag_chain.invoke({"input":msg})
    print("Response : ",response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
