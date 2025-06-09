from datasets import load_dataset
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

dataset = load_dataset("viber1/indian-law-dataset", split="train")
df = dataset.to_pandas()

df = df.rename(columns={"instruction": "Instruction", "output": "Response"})

documents = [
    Document(page_content=row["Response"], metadata={"question": row["Instruction"]})
    for _, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    persist_directory="legal_qa_vector_db"
)

print("✅ Vector DB created and saved.")
