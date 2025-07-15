import uvicorn
from pydantic import BaseModel,Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from app import MentalHealthAI


app = FastAPI()


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
    message: str


@app.post("/chat")
def chatbot_response(chat_input: ChatInput) -> Response:
    try:
        res = MentalHealthAI.invoke(HumanMessage(content=chat_input.message))
        return Response(response=str(res[-2].content), summary=str(res[-1].content), error="")
    except Exception as err:
        return Response(response="", summary="",error=str(err))
        



if __name__=="__main__":
    uvicorn.run(
        app="main:app",
        host="127.0.0.1",
        port=5000,
        reload=True,
    )