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

from Datachunksplitter import DataChunkSplitter
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)
groq_api_key = os.environ["GROQ_API_KEY"]
st.title("CardClarity: Credit Card Assistant")
st.sidebar.title("Enter CreditCard MainPage URLs")
urls = []
for i in range(3):
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
    retriever = vectorstore.as_retriever(
        # search_type="similarity", search_kwargs={"k": 3}
    )
    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with few sentences maximum.
    {context}
    Question: {input}
    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["input", "context"]
    )
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    result = retrieval_chain.invoke({"input": query})
    st.header("Answer")
    st.write(result["answer"])
    # Display sources, if available
    sources = result.get("sources", None)
    # sources = [doc.metadata["source"] for doc in docs]
    if sources:
        st.subheader("Sources:")
        # sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources:
            st.write(source)
