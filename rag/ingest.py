import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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