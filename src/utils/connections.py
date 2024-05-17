import os

from gen_ai_hub.proxy.core.proxy_clients import BaseProxyClient, get_proxy_client
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from hdbcli import dbapi
from langchain_community.vectorstores.hanavector import HanaDB

from src.configs import gen_ai_hub_service_key, hana_secrets


def _create_hana_connection() -> dbapi.Connection:
    """Create a connection to HANA DB.

    Returns:
        dbapi.Connection: Connection to HANA DB
    """
    return dbapi.connect(
        address=hana_secrets["host"],
        port=hana_secrets["port"],
        user=hana_secrets["user"],
        password=hana_secrets["password"],
        autocommit=True,
        sslTrustStore=hana_secrets["certificate"],
    )


def _get_gen_ai_hub_proxy_client() -> BaseProxyClient:
    """Get a Gen AI Hub proxy client.

    Returns:
        BaseProxyClient: Gen AI Hub proxy client
    """
    os.environ["AICORE_AUTH_URL"] = gen_ai_hub_service_key["url"]
    os.environ["AICORE_CLIENT_ID"] = gen_ai_hub_service_key["clientid"]
    os.environ["AICORE_CLIENT_SECRET"] = gen_ai_hub_service_key["clientsecret"]
    os.environ["AICORE_RESOURCE_GROUP"] = gen_ai_hub_service_key["appname"].split("!")[
        0
    ]
    os.environ["AICORE_BASE_URL"] = (
        f"{gen_ai_hub_service_key['serviceurls']['AI_API_URL']}/v2"
    )

    return get_proxy_client("gen-ai-hub")


def create_hana_vectordb_client(
    table_name: str, embedding_model_name: str = "text-embedding-ada-002"
) -> HanaDB:
    """Create a HanaDB client object from LangChain.

    Args:
        table_name (str): Name of HANA Vector table to work with
        embedding_model_name (str, optional): Name of embedding model to use for HANA Vector client.
            Defaults to "text-embedding-ada-002".

    Returns:
        HanaDB: HanaDB client object from LangChain
    """
    proxy_client = _get_gen_ai_hub_proxy_client()
    hana_conn = _create_hana_connection()
    embeddings = init_embedding_model(embedding_model_name, proxy_client=proxy_client)
    return HanaDB(embedding=embeddings, connection=hana_conn, table_name=table_name)
