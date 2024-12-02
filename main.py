import os
import streamlit as st
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import re
from Datachunksplitter import DataChunkSplitter
import json
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)
groq_api_key = os.environ["GROQ_API_KEY"]
st.title("CardClarity: Credit Card Assistant")
st.sidebar.title("Enter CreditCard MainPage URLs")
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",  # Provide the pre-trained model's path
    model_kwargs={"device": "cpu"},  # Pass the model configuration options
    encode_kwargs={"normalize_embeddings": False},  # Pass the encoding options
)
# Initialize an instance of ChatGroq with the mitral model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
if process_url_clicked:
    #################################################3
    ##DataLoading Phase
    urls = [url for url in urls if url]
    print("URLs are: ", urls)
    loader = WebBaseLoader(
        web_paths=urls,
        ###Specify the classname for dynamic webpages
        bs_kwargs=dict(
            # parse_only=bs4.SoupStrainer(class_=("group", "ArticleHeader-headline"))
        ),
    )
    print(loader)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    print("Data object length is: ", len(data))
    # Save the content of each document to a JSON file
    documents_to_save = []
    for i, doc in enumerate(data):
        documents_to_save.append(
            {
                "document_id": i,
                "content": doc.page_content,  # Assuming doc has a page_content attribute
                "metadata": doc.metadata,  # Assuming doc has a metadata attribute
            }
        )

    # Write the documents to a JSON file
    with open("data.json", "w") as json_file:
        json.dump(documents_to_save, json_file, indent=4)  # Save with pretty printing
    #################################################
    ##DataSplitting Phase
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    splitted_docs, sample_embedding = DataChunkSplitter().split_data_chunks(
        data, huggingface_embeddings=huggingface_embeddings
    )
    ##############################################
    ##VectorStore Phase
    if not huggingface_embeddings:
        raise ValueError("Embeddings are empty")
    vectorstore_openai = FAISS.from_documents(splitted_docs, huggingface_embeddings)
    print("dimention of faiss during storing", vectorstore_openai.index.d)
    print("Vectorstore processing is done.")
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    # Save the FAISS index(is library which can be work as vector database of fetched articles)
    vectorstore_openai.save_local("faiss_store")
    print("Vectorstore is saved to file.")
##########################################33
#############################################
##Question Answering Phase
query = main_placeholder.text_input(
    "What do you need help with regarding your credit card? "
)
if query:
    vectorstore = FAISS.load_local(
        "faiss_store",
        embeddings=huggingface_embeddings,
        allow_dangerous_deserialization=True,
    )
    print("dimention of loaded faiss", vectorstore.index.d)
    print("Vectorstore is loaded from file.")
    # Create a retriever to fetch relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},  # Retrieve more documents initially
    )

    # Retrieve relevant documents
    all_docs = retriever.get_relevant_documents(query)
    print("All docs are: ", len(all_docs))
    # Save all retrieved documents to a JSON file
    all_retrieved_documents = []
    for i, doc in enumerate(all_docs):
        all_retrieved_documents.append(
            {
                "document_id": i,
                "content": doc.page_content,  # Assuming doc has a page_content attribute
                "metadata": doc.metadata,  # Assuming doc has a metadata attribute
            }
        )

    # Write all retrieved documents to a JSON file
    with open("all_retrieved_documents.json", "w") as json_file:
        json.dump(all_retrieved_documents, json_file, indent=4)

    # Group documents by their metadata source
    grouped_docs = defaultdict(list)
    for doc in all_docs:
        source = doc.metadata.get("source")
        if source:
            grouped_docs[source].append(doc)

    # Select up to 2 documents from each unique source
    selected_documents = []
    for source, docs in grouped_docs.items():
        selected_documents.extend(docs[:2])  # Get up to 2 documents from each source

    # Save the selected documents to a JSON file
    retrieved_documents = []
    for i, doc in enumerate(selected_documents):
        retrieved_documents.append(
            {
                "document_id": i,
                "content": doc.page_content,  # Assuming doc has a page_content attribute
                "metadata": doc.metadata,  # Assuming doc has a metadata attribute
            }
        )

    # Write the selected documents to a JSON file
    with open("selected_documents.json", "w") as json_file:
        json.dump(retrieved_documents, json_file, indent=4)
    prompt_template = """STRICT CONTEXT-BASED ANSWER PROTOCOL:

    RULES:
    1. ONLY answer based EXCLUSIVELY on the provided context.
    2. If NO relevant information exists in the context, respond with: 
    "I cannot find a precise answer in the available context."
    3. STRICTLY PROHIBIT generating answers from external knowledge.
    4. If context is insufficient, suggest further research.

    CONTEXT:
    {context}

    QUERY: {input}

    RESPONSE GUIDELINES:
    - Be concise
    - Use information ONLY from the context
    - If uncertain, admit lack of definitive information

    Helpful Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["input", "context"]
    )
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    cleaned_query = re.sub(
        r"(explain this|tell me more|provide details|you know)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    result = retrieval_chain.invoke(
        {
            "input": cleaned_query,
            "max_tokens": 300,  # Limit response length
            "stop_sequences": ["External Knowledge:", "Beyond Context:"],
        }
    )
    st.header("Answer")
    st.write(result["answer"])
    # Display sources, if available
    sources = result.get("sources", None)
    sources = [doc.metadata["source"] for doc in docs]
    if sources:
        st.subheader("Sources:")
        # sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources:
            st.write(source)


# Enhanced Invocation with Additional Safeguards
def process_query(query):
    try:
        # Preprocess query to remove generic phrases
        cleaned_query = re.sub(
            r"(explain this|tell me more|provide details|you know)",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()

        result = retrieval_chain.invoke(
            {
                "input": cleaned_query,
                "max_tokens": 300,  # Limit response length
                "stop_sequences": ["External Knowledge:", "Beyond Context:"],
            }
        )

        return result["answer"]

    except Exception as e:
        return f"Error processing query: {str(e)}"
