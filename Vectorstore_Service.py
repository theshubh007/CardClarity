from langchain.retrievers import MergeRetriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Create separate vector stores for each URL
def create_multi_source_vectorstores(urls, data, huggingface_embeddings):
    vectorstores = []
    for url, doc_data in zip(urls, data):
        # Split documents for this specific URL
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=150
        )
        splitted_docs = text_splitter.split_documents([doc_data])

        # Create vector store for this URL
        vectorstore = FAISS.from_documents(splitted_docs, huggingface_embeddings)
        vectorstores.append(vectorstore)

    return vectorstores


# Enhanced Retrieval Method
def create_multi_source_retriever(vectorstores):
    retrievers = [
        vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 2,  # 2 documents per source
                "filter": lambda doc: doc.metadata["source"]
                == vs.index_to_docstore_id,  # Optional source filtering
            },
        )
        for vs in vectorstores
    ]

    return MergeRetriever(retrievers=retrievers)
