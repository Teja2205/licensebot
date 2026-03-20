import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "docs/"

def load_documents(docs_path=DOCS_PATH):
    documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_path, filename)
            loader = TextLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded: {filename}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content}")
        