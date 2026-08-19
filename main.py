from email.policy import strict
import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv 
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#page setup
st.set_page_config(page_title='PDF RAG Chatbot',
    page_icon=':books:',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('My PDF RAG Chatbot')
st.caption('Upload a PDF file and ask questions about it')

with st.sidebar:
    st.header('Settings')
    api_key=st.text_input('Enter your OpenAI API key',type='password',value=os.getenv('OPENAI_API_KEY',''))
    model_name=st.selectbox('Select the model',['gpt-4o','gpt-4o-mini'],index=0)
    chunk_size=st.slider('Select the chunk size',value=1000,min_value=100,max_value=2000,step=100)
    chunk_overlap=st.slider('Select the chunk overlap',value=200,min_value=50,max_value=300,step=10)
    top_k=st.slider('Retrived chunks',2,10,4)
if not api_key:
    st.info('Please enter your OpenAI API key to continue')
    st.stop()
os.environ['OPENAI_API_KEY']=api_key

# upload pdf file
st.subheader('Upload your PDF file')
uploaded_pdf=st.file_uploader('Upload your PDF file',type=['pdf'])


def build_vectorstore(pdf_bytes:bytes,size:int,overlap:int)->FAISS:
    """
    Build a vector store from a PDF file
    Args:
        pdf_bytes: bytes of the PDF file
        size: size of the chunks
        overlap: overlap of the chunks
    Returns:
        FAISS: vector store
    """
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp:
        temp.write(pdf_bytes)
        temp_path=temp.name
    try:
        docs=PyPDFLoader(temp_path).load()
    finally:
        os.unlink(temp_path)
    splitter=RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,  
            separators=['\n\n','\n','.',' ',''],
            )
    chunks=splitter.split_documents(docs)
    embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore=FAISS.from_documents(chunks,embeddings)
    return vectorstore
if uploaded_pdf is not None:
    pdf_sig=(uploaded_pdf.name,uploaded_pdf.size,chunk_size,chunk_overlap)
    if st.session_state.get('pdf_sig')!=pdf_sig:
        with st.spinner('Building vector store...'):
            st.session_state.vectorstore=build_vectorstore(uploaded_pdf.getvalue(),chunk_size,chunk_overlap)
            st.session_state.pdf_sig=pdf_sig
            st.write(st.session_state.vectorstore)
        st.success(f'Vector store for **{uploaded_pdf.name}** built successfully.Ask questions about the pdf')
        st.session_state.messages=[]
    else:
        st.info('Vector store already built')


load_dotenv(find_dotenv())

# RAG Prompt
RAG_PROMPT=ChatPromptTemplate.from_messages(
    [(
        "system",
         "You are a helpful assistant that you have to answer questions strictly using"
         "the context of the PDF file.\n"
         "Rules"
         "1:Answer only from the context of the pdf file"
         "2:If the is=nswer is not in your context just say:"
         "\"I could not find the answer to your question in the pdf file\".\n"
         "Cite the page number from the pdf file in the answer."
         "Context:\n{context}"
        ),
        ("human","{question}")
    ]
)



# st.write(RAG_PROMPT)

def format_docs(docs)->str:
    """Join retrived chunks, tagging each with its page number """
    return "\n\n".join(f"[page{d.metadata.get('page','?'),1}]{d.page_content} " for d in docs)
def get_chain(vectorstore:FAISS,model:str,k:int):
    retriever=vectorstore.as_retriever(search_kwargs={'k':k})
    llm=ChatOpenAI(model=model,temperature=0.5)
    return ({"context":retriever|format_docs,"question":RunnablePassthrough()}
    |RAG_PROMPT
    |llm
    |StrOutputParser()
    ),retriever

    # Check session state
if not 'vectorstore' in st.session_state:
    st.info('Please upload a PDF file to continue')
    st.stop()
if 'vectorstore' in st.session_state:
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
question=st.chat_input('Enter your question')
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.write(question)
    chain,retriever=get_chain(st.session_state.vectorstore,model_name,top_k)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response=chain.invoke(question)
            st.write(response)
    st.session_state.messages.append({'role':'assistant','content':response})
    

