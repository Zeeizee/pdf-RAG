
import os
import streamlit as st
from dotenv import load_dotenv,find_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from rag.ingest import build_vectorstore
from rag.chain import get_chain,as_sources,generate_overview,stream_answer


load_dotenv(find_dotenv(),override=True)


def is_streamlit_cloud() -> bool:
    return (
        os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower() == "cloud"
        or os.path.exists("/mount/src")
    )


def openai_error_message(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return (
            "OpenAI rejected this API key. It may be wrong, revoked, or not billed. "
            "Paste a valid key in the sidebar."
        )
    if isinstance(e, RateLimitError):
        return "OpenAI rate limit or quota hit. Check billing and try again."
    if isinstance(e, APIConnectionError):
        return "Could not reach OpenAI. Check your network and try again."
    if isinstance(e, APIError):
        return "OpenAI returned an error. Try again in a moment."
    return "Something went wrong while calling OpenAI. Try again."




#page setup
st.set_page_config(page_title='PDF RAG Chatbot',
    page_icon=':books:',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.title('PDF RAG Chatbot for Q/A')
st.caption('Upload a PDF file and ask questions about it')


with st.sidebar:
    st.header('Settings')
    api_key=st.text_input(
        'Enter your OpenAI API key',
        type='password',
        value='' if is_streamlit_cloud() else os.getenv('OPENAI_API_KEY', ''),
        placeholder='sk-...',
        help='On the public app, paste your own key. It stays in this browser only.',
    )
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
        try:
            with st.spinner('Building vector store...'):
                store = build_vectorstore(uploaded_pdf.getvalue(),chunk_size,chunk_overlap)
        except Exception as e:
            st.error(openai_error_message(e))
            st.stop()

        st.session_state.vectorstore = store
        st.session_state.pdf_sig = pdf_sig

        with st.spinner("Writing a short overview..."):
            try:
                st.session_state.overview = generate_overview(
                    st.session_state.vectorstore, model_name
                )
            except Exception:
                st.session_state.overview = None
                st.warning("Index ready, but overview could not be generated.")
        st.success(f"Vector store for **{uploaded_pdf.name}** built successfully.")
        st.session_state.messages = []


    # Check session state
if not 'vectorstore' in st.session_state:
    st.info('Please upload a PDF file to continue')
    st.stop()
if 'vectorstore' in st.session_state:
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
    overview = st.session_state.get("overview")
    if overview:
        st.subheader("Document overview")
        st.write(overview["summary"])
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for s in message["sources"]:
                    st.markdown(f"**page {s['page']}**")
                    st.write(s["snippet"])


question=st.chat_input('Ask your question')
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.write(question)
    chain,retriever=get_chain(st.session_state.vectorstore,model_name,top_k)
    docs = retriever.invoke(question)
    sources = as_sources(docs)
    try:
        with st.chat_message('assistant'):
            response = st.write_stream(stream_answer(docs, question, model_name))
            not_found = "could not find the answer" in response.lower()
            if sources and not not_found:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"**page {s['page']}**")
                        st.write(s["snippet"])
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'sources': [] if not_found else sources,
        })
    except Exception as e:
        st.error(openai_error_message(e))
    

