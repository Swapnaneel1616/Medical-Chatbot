from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List

#Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(data, 
                             glob="**/*.pdf", 
                             loader_cls=PyPDFLoader)


    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document])-> List[Document]:
    """Given a list of Document Objects, return a new list of Document Objects
        containing only source in metadata and the original page_content"""
    minimal_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        page_id = doc.metadata.get("page")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source":src , "id":page_id}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , 
                                                   chunk_overlap= 20)
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk


def download_embeddings():
    """Download and return the embedding model"""
    embedding = OpenAIEmbeddings()
    return embedding
