# Example of RAG Application with LangChain, HANA Vector DB and Generative AI Hub SDK

## About Repo
This repo shows an example of how a Python application doing Retrieval Augmented Generation (RAG) can be created using LangChain, HANA Vector DB and Generative AI Hub SDK.

This repo shows RAG codes in two different forms:
- A [Jupyer Notebook](https://github.com/yu-xuan-lee-sap/rag-langchain-hana-example/blob/44cd4e5001b8fa07e83ce304235f524cd55ea2a9/notebooks/rag_application_example.ipynb) which shows an example of RAG logic, from data ingestion to RAG
- A FastAPI application with a single endpoint performing RAG, accompanied with two Python scripts to index texts and delete a HANA Vector table

## Reproducing this Repo

### Prerequisites
- Python 3.9 and above [[link](https://www.python.org/downloads/)]
- Poetry 1.6.1 and above [[link](https://python-poetry.org/docs/#installing-with-the-official-installer)]
- \[Optional\] Python extension installed (if using VS Code)

### Cloning Repo
```bash
cd path/to/your/project/folder

# choose one of the options below to clone this repo

# clone via https
git clone https://github.com/yu-xuan-lee-sap/rag-langchain-hana-example.git

# clone via ssh
git clone git@github.com:yu-xuan-lee-sap/rag-langchain-hana-example.git
```

### Installing Dependencies
```bash
# for the case of multiple python installations in machine
poetry env use python3.x
# e.g. poetry env use python3.9

poetry install
```

### Starting FastAPI App
```bash
poetry shell
python run.py
```
The Swagger UI of the app can be accessed in the browser by navigating to http://localhost:8000/docs.

