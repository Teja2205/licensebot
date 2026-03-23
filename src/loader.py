import os
import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "docs/"

def load_txt_files(docs_path=DOCS_PATH):
    documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_path, filename)
            loader = TextLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded TXT: {filename}")
    return documents

def load_pdf_file(uploaded_file):
    """Load a PDF from a Streamlit uploaded file object"""
    documents = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_num + 1
                    }
                )
                documents.append(doc)
    print(f"Loaded PDF: {uploaded_file.name} ({len(documents)} pages)")
    return documents

def load_documents(docs_path=DOCS_PATH):
    """Load all .txt files from docs folder"""
    return load_txt_files(docs_path)

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks