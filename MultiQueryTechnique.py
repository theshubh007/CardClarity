from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_groq import ChatGroq
import os

groq_api_key = os.environ["GROQ_API_KEY"]


class MultiQueryTechnique:
    """
    This class that utilizes multi-query generation to enhance document retrieval from a vector database.
    It generates multiple perspectives of a user query to increase the relevance and diversity of retrieved
    documents, addressing limitations of standard distance-based similarity search.
    """

    def __init__(self, huggingface_embeddings):
        # Initialize with Hugging Face embeddings
        self.huggingface_embeddings = huggingface_embeddings

        # Multi Query: Different Perspectives
        template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Original question: {question}"""

        self.prompt_perspectives = ChatPromptTemplate.from_template(template)

        # Initialize the LLM (GrokLLM)
        self.llm =  ChatGroq(
            groq_api_key=groq_api_key, model_name="llama3-8b-8192"
        )  # Initialize with appropriate parameters if needed

    def generate_queries(self, question):
        """Generate alternative questions based on the input question."""
        # Generate responses using the LLM
        response = self.prompt_perspectives | self.llm | StrOutputParser()
        generated_questions = response.invoke({"question": question})

        # Split the generated questions into a list
        return generated_questions.split("\n")

    def get_unique_union(self, documents: list[list]):
        """Unique union of retrieved docs."""
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    def retrieve_documents(self, question, retriever):
        """Retrieve unique documents based on the generated queries."""
        queries = self.generate_queries(question)
        retrieval_chain = queries | retriever.map() | self.get_unique_union
        union_of_relevant_docs = retrieval_chain.invoke({"question": question})
        return union_of_relevant_docs
