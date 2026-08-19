from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



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
