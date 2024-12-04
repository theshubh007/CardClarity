import os
import time
import re
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from Datachunksplitter import DataChunkSplitter
from MultiQueryTechnique import MultiQueryTechnique


class CreditCardAssistant:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.environ["GROQ_API_KEY"]
        self.huggingface_embeddings = self.initialize_embeddings()
        self.llm = self.initialize_llm()
        self.retriever = None

    def initialize_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    def initialize_llm(self):
        return ChatGroq(groq_api_key=self.groq_api_key, model_name="mixtral-8x7b-32768")

    def process_urls(self, urls):
        urls = [url for url in urls if url]
        loader = WebBaseLoader(web_paths=urls)
        data = loader.load()
        self.save_documents(data)
        splitted_docs, _ = DataChunkSplitter().split_data_chunks(
            data, self.huggingface_embeddings
        )
        self.build_vectorstore(splitted_docs)

    def save_documents(self, data):
        documents_to_save = [
            {"document_id": i, "content": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(data)
        ]
        with open("data.json", "w") as json_file:
            json.dump(documents_to_save, json_file, indent=4)

    def build_vectorstore(self, splitted_docs):
        vectorstore = FAISS.from_documents(splitted_docs, self.huggingface_embeddings)
        vectorstore.save_local("faiss_store")
        self.retriever = vectorstore.as_retriever(search_type="similarity")

    def process_query(self, query):
        if not self.retriever:
            vectorstore = FAISS.load_local(
                "faiss_store",
                embeddings=self.huggingface_embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = vectorstore.as_retriever(search_type="similarity")

        all_docs = self.retriever.get_relevant_documents(query)
        self.save_retrieved_documents(all_docs)

        prompt_template = """STRICT CONTEXT-BASED ANSWER PROTOCOL:
        RULES:
        1. ONLY answer based EXCLUSIVELY on the provided context.
        2. If NO relevant information exists in the context, respond with: "I cannot find a precise answer in the available context."
        3. STRICTLY PROHIBIT generating answers from external knowledge.
        4. If context is insufficient, suggest further research.
        CONTEXT: {context}
        QUERY: {input}
        RESPONSE GUIDELINES:
        - Be concise
        - Use information ONLY from the context
        - If uncertain, admit lack of definitive information
        Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["input", "context"]
        )

        cleaned_query = re.sub(
            r"(explain this|tell me more|provide details|you know)",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()
        multi_query_technique = MultiQueryTechnique(self.huggingface_embeddings)
        alternative_queries = multi_query_technique.generate_queries(query)
        print("List of Alternative queries to make context more accurate:")
        for alt_query in alternative_queries:
            print("Alt query:", alt_query)

        all_retrieved_docs = []
        for alt_query in alternative_queries:
            retrieved_docs = self.retriever.invoke(alt_query)
            all_retrieved_docs.append(retrieved_docs)    

        # Get unique documents from all retrieved documents
        unique_docs = multi_query_technique.get_unique_union(all_retrieved_docs)    


        document_chain = create_stuff_documents_chain(self.llm, PROMPT)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        result = retrieval_chain.invoke(
            {
                "input": cleaned_query,
                "max_tokens": 300,
                "stop_sequences": ["External Knowledge:", "Beyond Context:"],
                # "context": unique_docs,
            }
        )
        # unique_docs = [doc for doc in unique_docs if doc.page_content]

        return result["answer"], result.get("sources", None)

    def save_retrieved_documents(self, docs):
        retrieved_documents = [
            {"document_id": i, "content": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        with open("all_retrieved_documents.json", "w") as json_file:
            json.dump(retrieved_documents, json_file, indent=4)
