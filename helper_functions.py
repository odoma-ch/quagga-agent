import torch
from tqdm import tqdm
from chonkie import TokenChunker
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

from langchain_community.embeddings import InfinityEmbeddings

# tried with this but doesnt perform well: https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R
# currently using this: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
# MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# embedding_model = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME, 
#     model_kwargs={"trust_remote_code": True, "device": device}, 
#     encode_kwargs={"padding_side": "left", "device": device}
# )
# reference: https://python.langchain.com/docs/integrations/text_embedding/infinity/
infinity_api_url = "https://qwen-embedding-route-graphia-app1-staging.apps.bst2.paas.psnc.pl"
embedding_model = InfinityEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B", infinity_api_url=infinity_api_url
)

chunker = TokenChunker(chunk_size=1024, chunk_overlap=50, tokenizer="gpt2")

def load_shex_file(file_path: str) -> str:
    if not file_path.endswith(".ttl"):
        raise Exception("File is not a ShEx file")
    loader = TextLoader(file_path)
    return loader.load()[0].page_content


def chunk_shex_file(shex_content: str) -> tuple[str, list[Document], list[str]]:
    shex_content_split = shex_content.split("\n\n")
    shex_prefixes = shex_content_split[0]

    shex_schema = shex_content_split[1].strip().rstrip()
    # split by the spacing and nextline from the void-generator
    shex_schema_split = shex_schema.split("\n                \n")

    doc_chunks = []
    chunk_titles = []
    for index, chunk in enumerate(shex_schema_split):
        if "{" in chunk:
            title = chunk.split("{")[0].strip()
            chunk_titles.append(title)
        doc_chunks.append(
            Document(page_content=chunk, metadata={"chunk_index": index}))

    return shex_prefixes, doc_chunks, chunk_titles


def chunk_sample_file(sample_content: str) -> tuple[str, list[Document], list[str]]:
    sample_content_split = sample_content.split("\n\n")
    sample_prefixes = sample_content_split[0]
    sample_rows = sample_content_split[1:]

    doc_chunks = []
    for index, chunk in enumerate(sample_rows):
        doc_chunks.append(
            Document(page_content=chunk, metadata={"chunk_index": index}))
    return sample_prefixes, doc_chunks


def get_embeddings_for_chunks(collection_name: str, doc_chunks: list[Document], use_chonkie: bool = False):
    if use_chonkie:
        original_count = len(doc_chunks)
        combined_content = "\n\n".join([doc.page_content for doc in doc_chunks])
        chunks = chunker.chunk(combined_content)
        
        # Create new optimized chunks
        optimized_chunks = []
        for index, chunk in enumerate(chunks):
            optimized_chunks.append(Document(
                page_content=chunk.text,
                metadata={"chunk_id": index, "original_chunk_count": original_count}
            ))
        
        doc_chunks = optimized_chunks
        print(f"Original chunks: {original_count}, Optimized chunks: {len(doc_chunks)}")

    batch_size = 128
    vector_store = None
    with tqdm(total=len(doc_chunks), desc="Processing documents") as pbar:
        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i + batch_size]
            
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    collection_name=collection_name,
                    documents=batch,
                    embedding=embedding_model,
                    persist_directory="./chroma_db"
                )
            else:
                vector_store.add_documents(batch)
            
            pbar.update(len(batch))

    return vector_store


def multiline_to_single(text, separator=' '):
    return separator.join(line.strip() for line in text.strip().split('\n') if line.strip())
