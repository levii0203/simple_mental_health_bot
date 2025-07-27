from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
load_dotenv()


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

#Chain for reflecting back generator's response with a review
Reflective_Chain = (
    chat_input
    | llm
)

