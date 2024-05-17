from pydantic import BaseModel, Field


class RAGRequestBody(BaseModel):
    query: str = Field(
        description="The query to generate responses for",
        example="What is Python typically used for?",
    )

    hana_vector_table_name: str = Field(
        description="The name of the HANA vector table to use for retrieval",
        default="RAG_EXAMPLE_VECTORSTORE",
    )
    llm_model_name: str = Field(
        description="The name of the LLM model to use for generation", default="gpt-4"
    )
    embedding_model_name: str = Field(
        description="The name of the embedding model to use for retrieval",
        default="text-embedding-ada-002",
    )
