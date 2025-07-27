from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq (
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_retries=2,
    timeout=None
)

#system_role 
system_role = SystemMessagePromptTemplate.from_template(
    template = [''' 
        You are an assistant therapist, a mental health professional and a counselor too. Your job is to consult & give therapy in response to user prompts. 
        Make sure your behavior is like human and user should not feel if it's talking to an AI.
        Analyze user's mental health and give the user practical solutions in your therapy session. Also suggest any pill/drug/medicine if user needs it.
        You are in a conversation as a therapist and mental health expert , give response befitted to the role.
        Only respond in plain english , no symbols or digits must be used.
        
        To provide context-aware responses, you're provided with the context of the chat. If context is not given, that means it's the start of the conversation.
        You are provided with some critiques as well , response with a revised version of your previous attempts.
        {reflection}
        '''
    ]

)

#human_role
human_role = HumanMessagePromptTemplate.from_template(
    template=[
        #Context for past history retrieved from chroma
        '''
        Context:{context}
        Client message:{user_input}
        '''
    ],
)

#input_template
chat_input = ChatPromptTemplate.from_messages([system_role,human_role])

#chain for generating response to client's message
Generator_Chain = (
    chat_input
    | llm
)







