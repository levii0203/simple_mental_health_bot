import os
import logging
from chains.generator_chain import Generator_Chain
from chains.reflector_chain import Reflective_Chain
from chains.summarizer_chain import Summarizer_Chain
from langchain_core.messages import AIMessage
from langgraph.graph import MessageGraph, END
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chroma import Chroma_store , retrieve_context


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("mentalhealthai")

from dotenv import load_dotenv
load_dotenv()

DB_URI = os.getenv("DB_URI")

#Message graph used here with default state
graph = MessageGraph()


def GenerativeAgent(messages):
    """_summary_

    Args:
        messages (list[str]): Messagegraph state

    Returns:
        AIMessage: Response to client's message 
    """
    logger.info("Generating response")
    #client's message
    user_input = messages[0].content
    #session id
    session_id = messages[0].additional_kwargs.get("session_id",0)
    #reflected message from reflector chain
    reflection = ""
    #context fetched from postgres
    context = ""
    if len(messages)>1:
        """Retrieving inputs for generator chain"""
        user_input = messages[len(messages)-(len(messages)%6)].content
        reflection = messages[-1].content
        
        #retrieving from chat history embedding
        context = retrieve_context(user_input=user_input,session_id=session_id)
    else:
        #Fetching chat history
        chat_history = PostgresChatMessageHistory(connection_string=DB_URI,session_id=session_id)
        chat_history.add_user_message(user_input)
        
        #splitting history into docs for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        messages = chat_history.messages
        texts = [message.content for message in messages]
        doc = text_splitter.create_documents(texts, metadatas=[{"session_id": session_id} for _ in texts])
        docs = text_splitter.split_documents(doc)
        
        #deleting pre existing chat history documents for the same session
        all_docs = Chroma_store.get()
        session_ids = [id for id in all_docs["ids"] if id.startswith(session_id)]
        if session_ids:
            Chroma_store.delete(ids=session_ids)
        
        #each doc for a particular session id a unique id 
        doc_ids = [f"{session_id}_{i+1}" for i in range(len(docs))]
        Chroma_store.add_documents(
            docs,
            ids=doc_ids
        )
    res = Generator_Chain.invoke({"user_input":user_input,"reflection":reflection,"context":context})
    if len(messages)>1:
        #setting iteration for the first response
        cnt = messages[-1].additional_kwargs.get("iteration",0)
        return [AIMessage(content=res.content,additional_kwargs={"iteration":cnt+1})]
    return  [AIMessage(content=res.content,additional_kwargs={"iteration":1})]


def ReflectiveAgent(messages):
    """_summary_

    Args:
        messages (_type_): Messagegraph state

    Returns:
        AIMessage: Reviews to Generator's response
    """
    
    logger.info("Reflecting response")
    cnt = messages[-1].additional_kwargs.get("iteration",0)
    res = Reflective_Chain.invoke(messages[-1].content)
    return [AIMessage(content=res.content,additional_kwargs={"iteration":cnt+1})]


def CheckIteration(messages):
    """Checking number of iterations"""
    
    logger.info("Checking iterations")
    cnt = messages[-1].additional_kwargs.get("iteration",0)
    if cnt>4:
        #Ending if the generator-reflector loop happened twice
        session_id = messages[0].additional_kwargs.get("session_id",0)
        chat_history = PostgresChatMessageHistory(connection_string=DB_URI,session_id=session_id)
        chat_history.add_ai_message(messages[-1].content)
        return "END"
    return "reflect"


"""Not necessary to use!!!"""
"""TO BE IGNORE"""
def SummarizingAgent(messages):
    print("summarizing")
    user_input = messages[0].content
    therapist_response = messages[-1].content 
    res = Summarizer_Chain.invoke({"user_input":user_input,"response":therapist_response})
    return [AIMessage(content=res.content)]


graph.add_node("generate",GenerativeAgent)
graph.add_node("check",CheckIteration)
graph.add_node("reflect",ReflectiveAgent)

graph.set_entry_point("generate")
graph.add_conditional_edges(
    "generate",
    CheckIteration,
    {"reflect":"reflect", "END":END}
    
)

graph.add_edge("reflect","generate")


MentalHealthAI = graph.compile()




