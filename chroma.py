from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#hauggingFace embedding model
embeddings = HuggingFaceEmbeddings(model="intfloat/e5-small-v2")

Chroma_store = Chroma(
    collection_name="chat_history_vectors",
    embedding_function=embeddings,
    persist_directory="./chroma_db" 
)

#context retrieval for generator_chain
def retrieve_context(user_input:str,session_id:str):
    """_summary_

    Args:
        user_input (str): client's message
        session_id (str): session id

    Returns:
        string: retrieved data from stored chat history
    """
    results = Chroma_store.similarity_search(
        user_input,
        k=3,
        filter={"session_id": session_id}
    )
    return "\n---\n".join([doc.page_content for doc in results])