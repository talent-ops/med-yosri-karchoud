import os

os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


@st.cache_data
def get_customers():
    client = QdrantClient(url=QDRANT_URL)
    points, _ = client.scroll(COLLECTION_NAME, limit=5000)
    names = set()
    for point in points:
        names.add(point.payload["metadata"]["bill_to"])
    return sorted(names)


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Answer the question using only the context.
    If you don't know, say i don't know.

    question:
    {question}

    context:
    {context}
    """
)

st.title("Invoices search")

customer = st.sidebar.selectbox("Bill To", ["All"] + get_customers())
min_total = st.sidebar.number_input("Minimum total", value=0.0)
k = st.sidebar.slider("Number of results", 1, 20, 5)

question = st.text_input("Ask something about the invoices")

if question:
    conditions = [FieldCondition(key="metadata.total", range=Range(gte=min_total))]
    if customer != "All":
        conditions.append(FieldCondition(key="metadata.bill_to", match=MatchValue(value=customer)))

    documents = get_vector_store().similarity_search(question, k=k, filter=Filter(must=conditions))

    if len(documents) == 0:
        st.write("No invoice found")
    else:
        table = []
        for doc in documents:
            table.append(doc.metadata)
        st.table(table)

        context = "\n\n".join([doc.page_content for doc in documents])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})
        st.write(answer)
