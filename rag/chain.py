from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json


RAG_PROMPT=ChatPromptTemplate.from_messages(
    [(
        "system",
         "You are a helpful assistant that you have to answer questions strictly using"
         "the context of the PDF file.\n"
         "Rules"
         "1:Answer only from the context of the pdf file"
         "2:If the answer is not in your context just say:"
         "\"I could not find the answer to your question in the pdf file\".\n"
         "Cite the page number from the pdf file in the answer."
         "Context:\n{context}"
        ),
        ("human","{question}")
    ]
)

OVERVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You help a reader get oriented in a PDF.\n"
            "Use only the context. Do not invent facts.\n"
            "Reply with JSON only, no markdown:\n"
            '{{"summary": "2-4 sentences"}}\n'
            "Context:\n{context}",
            
        ),
        ("human", "Summarize this PDF gracefully as it may convey a lot of information."),
    ]
)

def generate_overview(vectorstore: FAISS, model: str):
    docs = vectorstore.similarity_search("What is this document about?", k=8)
    llm = ChatOpenAI(model=model, temperature=0.2)
    raw = (OVERVIEW_PROMPT | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs)}
    )
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    questions = [q.strip() for q in data.get("questions", []) if q.strip()]
    return {
        "summary": (data.get("summary") or "").strip(),
        "questions": questions[:5],
    }

def page_label(doc):
    page = doc.metadata.get("page", 0)
    if isinstance(page, int):
        return page + 1
    return page

def as_sources(docs):
    sources = []
    for d in docs:
        snippet = (d.page_content or "").strip().replace("\n", " ")
        sources.append({
            "page": page_label(d),
            "snippet": snippet[:400],
        })
    return sources


def format_docs(docs)->str:
    """Join retrived chunks, tagging each with its page number """
    return "\n\n".join(f"[page{page_label(d)}]{d.page_content} " for d in docs)


def get_chain(vectorstore:FAISS,model:str,k:int):
    retriever=vectorstore.as_retriever(search_kwargs={'k':k})
    llm=ChatOpenAI(model=model,temperature=0.5)
    return ({"context":retriever|format_docs,"question":RunnablePassthrough()}
    |RAG_PROMPT
    |llm
    |StrOutputParser()
    ),retriever

def stream_answer(docs, question, model):
    llm = ChatOpenAI(model=model, temperature=0.5)
    chain = RAG_PROMPT | llm | StrOutputParser()
    payload = {"context": format_docs(docs), "question": question}
    for piece in chain.stream(payload):
        yield piece
