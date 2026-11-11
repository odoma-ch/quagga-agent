import torch
from tqdm import tqdm
from rdflib import Graph
from chonkie import TokenChunker
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from tools import ShexRetrieverTool
import chromadb
import requests
import sys

chunker = TokenChunker(
    chunk_size=1024, chunk_overlap=50, tokenizer="Qwen/Qwen3-Embedding-0.6B"
)


def load_shex_file(file_path: str) -> str:
    if not file_path.endswith(".ttl"):
        raise Exception("File is not a ShEx file")
    loader = TextLoader(file_path)
    return loader.load()[0].page_content


def chunk_shex_file(shex_content: str) -> tuple[str, list, list]:
    shex_content_split = shex_content.split("\n\n")
    shex_prefixes = shex_content_split[0]

    shex_schema = shex_content_split[1].strip().rstrip()
    # split by the spacing and nextline from the void-generator
    shex_schema_split = shex_schema.split("\n                \n")

    doc_chunks = []
    chunk_titles = []
    for index, chunk in enumerate(shex_schema_split):
        if not chunk.strip():
            continue
        if "{" in chunk:
            title = chunk.split("{")[0].strip()
            chunk_titles.append(title)
        doc_chunks.append(Document(page_content=chunk, metadata={"chunk_index": index}))

    return shex_prefixes, doc_chunks, chunk_titles


def load_sample_file(file_path: str) -> str:
    if not file_path.endswith(".nt"):
        raise Exception("File is not a sample file")
    loader = TextLoader(file_path)
    return loader.load()[0].page_content


def chunk_sample_file(sample_content: str) -> list:
    sample_content_split = sample_content.split("\\n")
    doc_chunks = []
    for index, chunk in enumerate(sample_content_split):

        try:
            g = Graph()
            g.parse(data=chunk, format="nt")
            s, p, o = next(iter(g))
            chunk_content = f"{s} {p} {o}"
        except StopIteration:
            print(f"Warning: Chunk {index} is empty or invalid, skipping")
            continue

        doc_chunks.append(
            {"page_content": chunk_content, "metadata": {"chunk_index": index}}
        )
    return doc_chunks


def chunk_void_file(void_content: str) -> tuple[str, list]:
    void_content_split = void_content.split("\\n\\n")
    void_prefixes = void_content_split[0]

    doc_chunks = []
    for index, chunk in enumerate(void_content_split[1:]):
        doc_chunks.append({"page_content": chunk, "metadata": {"chunk_index": index}})

    return void_prefixes, doc_chunks


def get_embeddings_for_chunks(
    collection_name: str,
    doc_chunks: list,
    embedding_model,
    client=None,
    use_chonkie: bool = False,
):
    if use_chonkie:
        original_count = len(doc_chunks)
        combined_content = "\\n\\n".join(
            [
                doc.get("page_content") if isinstance(doc, dict) else doc.page_content
                for doc in doc_chunks
            ]
        )
        chunks = chunker.chunk(combined_content)

        optimized_chunks = []
        for index, chunk in enumerate(chunks):
            optimized_chunks.append(
                {"page_content": chunk.text, "metadata": {"chunk_id": index}}
            )

        doc_chunks = optimized_chunks
        print(f"Original chunks: {original_count}, Optimized chunks: {len(doc_chunks)}")

    langchain_docs = []
    for doc in doc_chunks:
        if isinstance(doc, dict):
            langchain_docs.append(
                Document(
                    page_content=doc["page_content"], metadata=doc.get("metadata", {})
                )
            )
        else:
            langchain_docs.append(doc)

    batch_size = 128
    vector_store = None
    with tqdm(total=len(langchain_docs), desc="Processing documents") as pbar:
        for i in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[i : i + batch_size]

            if vector_store is None:
                vector_store = Chroma.from_documents(
                    collection_name=collection_name,
                    documents=batch,
                    embedding=embedding_model,
                    client=client,
                )
            else:
                vector_store.add_documents(batch)

            pbar.update(len(batch))

    return vector_store


def slugify_kg_name(kg_name: str) -> str:
    return kg_name.lower().replace(" ", "_")


def fetch_kg_metadata(kg_name: str, kg_endpoint_url: str, metadata_url: str) -> dict:
    """Fetch KG metadata from the metadata endpoint."""
    response = requests.get(
        metadata_url,
        params={"kg_name": kg_name, "kg_endpoint_url": kg_endpoint_url},
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    return {
        "kg_name": kg_name,
        "endpoint_url": kg_endpoint_url,
        "shex_content": data.get("shex_content", ""),
    }


def setup_kg_rag_environment(
    kg_name: str,
    kg_endpoint_url: str,
    metadata_url: str,
    client: chromadb.PersistentClient,
    embedding_model,
    k: int = 3,
):
    """Setup RAG environment for a specific KG: fetch metadata, store in ChromaDB, return retriever."""
    metadata = fetch_kg_metadata(kg_name, kg_endpoint_url, metadata_url)
    shex_content = metadata["shex_content"]

    collection_name = slugify_kg_name(kg_name) + "_shex"
    available_collections = [c.name for c in client.list_collections()]

    if collection_name not in available_collections:
        shex_prefixes, shex_chunks, chunk_titles = chunk_shex_file(shex_content)
        shex_vector_store = get_embeddings_for_chunks(
            collection_name,
            shex_chunks,
            embedding_model,
            client=client,
            use_chonkie=False,
        )
        retriever_tool = ShexRetrieverTool(vectorstore=shex_vector_store, k=k)
    else:
        retriever_tool = ShexRetrieverTool(
            vectorstore=Chroma(
                collection_name=collection_name,
                client=client,
                embedding_function=embedding_model,
            ),
            k=k,
        )
        shex_prefixes, _, chunk_titles = chunk_shex_file(shex_content)

    return retriever_tool, shex_prefixes, chunk_titles, metadata["endpoint_url"]


def multiline_to_single(text, separator=" "):
    return separator.join(
        line.strip() for line in text.strip().split("\n") if line.strip()
    )
