import argparse

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

from src.utils.connections import create_hana_vectordb_client


def main(args):
    table_name = args.table_name
    input_csv_filepath = args.input_csv_filepath
    page_content_column_name = args.page_content_column_name

    # Create the table
    print(f"Creating table {table_name} if it doesn't exist...")
    hana_vectordb = create_hana_vectordb_client(
        table_name=table_name, embedding_model_name="text-embedding-ada-002"
    )
    input_df = pd.read_csv(input_csv_filepath)
    loader = DataFrameLoader(
        data_frame=input_df, page_content_column=page_content_column_name
    )
    documents_to_index = loader.load()
    hana_vectordb.add_documents(documents_to_index)

    print(
        f"{len(documents_to_index)} documents written to table {table_name} created successfully"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and write vectors to a HANA vector table"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        help="The name of the table to write to",
        default="RAG_EXAMPLE_VECTORSTORE",
    )
    parser.add_argument(
        "--input-csv-filepath",
        type=str,
        help="The filepath of the input CSV file",
        default="data/rag_example_inputs.csv",
    )
    parser.add_argument(
        "--page-content-column-name",
        type=str,
        help="The name of the column containing the page content",
        default="text",
    )
    args = parser.parse_args()

    main(args)
