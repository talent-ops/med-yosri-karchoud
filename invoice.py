import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"
FOLDER = Path("data/1000+ PDF_Invoice_Folder")

documents = []

for pdf in FOLDER.glob("*.pdf"):
    page = PyPDFLoader(str(pdf)).load()[0]
    text = page.page_content
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if "Ship To" not in lines:
        continue

    code = re.search(r"# (\d+)", text).group(1)
    bill_to = lines[lines.index("Bill To") + 2]

    i = lines.index("Ship To") + 2
    ship_to = ""
    while not re.match(r"^[A-Z][a-z]{2} \d{2} \d{4}$", lines[i]):
        ship_to = ship_to + " " + lines[i]
        i = i + 1
    ship_to = ship_to.strip()

    prices = re.findall(r"\$([\d,]+\.\d{2})", text)
    prices = [float(p.replace(",", "")) for p in prices]

    if "Discount" in text:
        amount, discount, shipping, total = prices[-4:]
    else:
        amount, shipping, total = prices[-3:]
        discount = 0.0

    metadata = {
        "invoice_code": code,
        "bill_to": bill_to,
        "ship_to": ship_to,
        "amount": amount,
        "discount": discount,
        "shipping": shipping,
        "total": total,
    }
    documents.append(Document(page_content=text, metadata=metadata))

print(len(documents), "invoices loaded")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True,
)

print("Invoices stored in Qdrant")
