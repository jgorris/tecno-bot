import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


st.header(":robot_face: TecnoBot :gear:", divider='rainbow')
st.write("""Bienvenido al chatbot de la asignatura de tecnologia, espero servirte de ayuda para 
    responder a todas las preguntas relacionadas con la asignatura.""")
st.sidebar.success("")
st.write("---")
OPENAI_API_KEY = st.text_input('Introduce tu OpenAI API Key', type='password')
st.write("Si aun no tienes una API Key,[pulsa aquí](https://platform.openai.com/api-keys)")
pdf_obj = st.file_uploader("Carga tu documento PDF", type="pdf", on_change=st.cache_resource.clear)

#evita voler a realizar la funcion embeddings si posee el mismo valor que la ultima vez ejecutada
@st.cache_resource
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_input("Haz una pregunta sobre tu PDF:", disabled= not(pdf_obj and OPENAI_API_KEY))

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3) #extraemos los 3 trozos más relevantes del pdf
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)
        prompt_template = """
            Eres un asistente que debe simular a un profesor. 
            Para ayudar a los usuarios, haz uso de la información que se da en el texto
            para responder a las preguntas. Si no sabes la respuesta, 
            simplemente di que no la sabes, no intentes inventar una respuesta.
            
            Context: {context}

            Question: {question}
            
            Respuesta util:
            """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)    