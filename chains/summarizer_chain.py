from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


load_dotenv()


llm = ChatGroq (
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2,
    timeout=None
)


system_role = SystemMessagePromptTemplate.from_template(
    template = [''' 
        You are an assistant summarizer. You are given a conservation between a therapist and his client. 
        Client speaks first. Summarize the conversation in such a way that therapist can refer back to it to 
        get the full context of their earlier session. Don't use symbols, digits & any ascii characters except plain english in your response.
        Refer therapist as the second person.
        '''
    ]

)

human_role = HumanMessagePromptTemplate.from_template(
    template=[
        '''
        client:  {user_input}
        therapist_response: {response}
        '''
    ],
)

chat_input = ChatPromptTemplate.from_messages([system_role,human_role])
\
Summarizer_Chain = (
    chat_input
    | llm
)