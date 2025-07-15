from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate


load_dotenv()


'''class GeneratorResponse(BaseModel):
    """Always use this tool to structure your response to the user."""
    user_input: str = Field(description="This is the given user input.")
    message: str = Field(description="the response message should be here")
    medication: Optional[str] = Field(description="optional field if you ever had to suggest a list of medicines")'''
    

llm = ChatGroq (
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2,
    timeout=None
)



system_role = SystemMessagePromptTemplate.from_template(
    template = [''' 
        You are an assistant therapist, a mental health professional and a counselor too. Your job is to consult & give therapy in response to user prompts. 
        Make sure your behavior is like human and user should not feel if it's talking to an AI.
        Analyze user's mental health and give the user practical solutions in your therapy session. Also suggest any pill/drug/medicine if user needs it.
        You are in a conversation as a therapist and mental health expert , give response befitted to the role.
        Only respond in plain english , no symbols or digits must be used.
        
        You are provided with some critiques as well , response with a revised version of your previous attempts.
        {reflection}
        '''
    ]

)

human_role = HumanMessagePromptTemplate.from_template(
    template=[
        '''
        {user_input}
        '''
    ],
)

chat_input = ChatPromptTemplate.from_messages([system_role,human_role])

Generator_Chain = (
    chat_input
    | llm
)

