from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.language_models import BaseLanguageModel

from src.api.models.requests import RAGRequestBody
from src.utils.connections import (
    _get_gen_ai_hub_proxy_client,
    create_hana_vectordb_client,
)


def init_hana_vectordb_client(request_body: RAGRequestBody) -> HanaDB:
    """Initialize HanaDB client in the context of a FastAPI request.

    Args:
        request_body (RAGRequestBody): Request body containing the HANA table name and embedding model name

    Returns:
        HanaDB: Initialized HanaDB client
    """
    return create_hana_vectordb_client(
        table_name=request_body.hana_vector_table_name,
        embedding_model_name=request_body.embedding_model_name,
    )


def init_llm_client(request_body: RAGRequestBody) -> BaseLanguageModel:
    """Initialize a language model client in the context of a FastAPI request.

    Args:
        request_body (RAGRequestBody): Request body containing the LLM model name

    Returns:
        BaseLanguageModel: Initialized language model client
    """
    return init_llm(
        request_body.llm_model_name, proxy_client=_get_gen_ai_hub_proxy_client()
    )
