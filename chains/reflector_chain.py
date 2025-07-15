from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


load_dotenv()

'''class ReflectorResponse(BaseModel):
    """Always use this tool to structure your response to the user."""
    reflection: str = Field(description="This is your response here")'''


llm = ChatGroq (
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2,
    timeout=None
)


system_role = SystemMessagePromptTemplate.from_template(
    template = [''' 
        You are an assistant therapist watcher. The user is a therapist and a mental health expert.
        The user has needs a feedback on his response to one of his clients in a therapy session.
        Give your feedback and critique on his therapy response. Only respond in plain english , no symbols or digits must be used.
        Remind the therapist that he is a mental health professional because he tell his clients to go to mental health professional/expert/pychiatrist.
        Suggest improvements if needed.
        '''
    ]

)

human_role = HumanMessagePromptTemplate.from_template(
    '''
    This is my response to my client
    {message}
    '''
)


chat_input = ChatPromptTemplate.from_messages([system_role,human_role])

Reflective_Chain = (
    chat_input
    | llm
)

