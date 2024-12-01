import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataChunkSplitter:
    def __init__(self):
        pass

    def split_data_chunks(self, data, huggingface_embeddings):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=150
        )
        splitted_docs = text_splitter.split_documents(data)
        print("Splitted chunks are: ", splitted_docs)
        sample_embedding = np.array(
            huggingface_embeddings.embed_query(splitted_docs[0].page_content)
        )
        print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of the embedding: ", sample_embedding.shape)
        return splitted_docs, sample_embedding
