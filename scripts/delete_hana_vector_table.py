import argparse

from src.utils.connections import create_hana_vectordb_client


def main(args):
    table_name = args.table_name

    # Delete the table
    print(f"Deleting table {table_name}...")
    hana_vectordb = create_hana_vectordb_client(
        table_name=table_name, embedding_model_name="text-embedding-ada-002"
    )
    hana_vectordb.delete(filter={})

    print(f"Table {table_name} deleted successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a HANA vector table")
    parser.add_argument(
        "--table-name",
        type=str,
        help="The name of the table to delete",
        default="RAG_EXAMPLE_VECTORSTORE",
    )
    args = parser.parse_args()

    main(args)
