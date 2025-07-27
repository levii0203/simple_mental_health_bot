import os
import uvicorn
import logging
from pydantic import BaseModel,Field
from fastapi import FastAPI , logger
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from chroma import Chroma_store
from app import MentalHealthAI

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["Content-Type","Authorization"],
    allow_methods = ["GET","POST"]
)

class Response(BaseModel):
    response:str = Field(description="chatbot_response")
    summary:str = Field(description="summary of the conversation")
    error:str = Field(description="response_error")


class ChatInput(BaseModel):
    #Model for invoking mental health agent
    message: str
    session_id: str


@app.post("/chat")
def chatbot_response(chat_input: ChatInput) -> Response:
    try:
        res = MentalHealthAI.invoke(HumanMessage(content=chat_input.message,additional_kwargs={"session_id":chat_input.session_id}),
        config = {"configurable": {"thread_id": chat_input.session_id}})
        logger.info(f"MentalHealth_Agent Response: {res[-1].content}")
        
        return Response(response=str(res[-1].content), summary="", error="")
    except Exception as err:
        return Response(response="", summary="",error=str(err))
        

if __name__=="__main__":
    uvicorn.run(
        app="main:app",
        host="127.0.0.1",
        port=5000,
        reload=True,
    )