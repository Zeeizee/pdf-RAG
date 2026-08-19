
import os
import streamlit as st
from dotenv import load_dotenv,find_dotenv 
from rag.ingest import build_vectorstore
from rag.chain import get_chain



#page setup
st.set_page_config(page_title='PDF RAG Chatbot',
    page_icon=':books:',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('My PDF RAG Chatbot for Q/A')
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
    

