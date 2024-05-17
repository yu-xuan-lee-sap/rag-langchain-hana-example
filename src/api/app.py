from fastapi import Depends, FastAPI
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.api.models.requests import RAGRequestBody
from src.api.models.responses import RAGResponseBody
from src.utils.app import init_hana_vectordb_client, init_llm_client

app = FastAPI(
    title="Example RAG API",
    description="An example REST API that does RAG using LangChain, HANA Vector DB and Generative AI Hub SDK",
    version="0.1.0",
)


@app.post("/rag", response_model=RAGResponseBody)
async def run_rag_chain(
    request_body: RAGRequestBody,
    hana_vectordb: HanaDB = Depends(init_hana_vectordb_client),
    llm: BaseLanguageModel = Depends(init_llm_client),
):
    """Run a RAG chain to generate responses for the given query using the given HANA vector table, LLM model and
    embedding model."""
    hana_vector_retriever = hana_vectordb.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {
            "context": hana_vector_retriever,
            "question": RunnablePassthrough(),
        }
    )
    rag_chain = setup_and_retrieval | prompt | llm | output_parser
    rag_response = rag_chain.invoke(request_body.query)
    return {"response": rag_response}
