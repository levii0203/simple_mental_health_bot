from chains.generator_chain import Generator_Chain
from chains.reflector_chain import Reflective_Chain
from chains.summarizer_chain import Summarizer_Chain
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import MessageGraph, END


graph = MessageGraph()


def GenerativeAgent(messages):
    print("generate")
    user_input = messages[0].content
    reflection = messages[-1].content
    res = Generator_Chain.invoke({"user_input":user_input,"reflection":reflection})
    return [AIMessage(content=res.content)]

def ReflectiveAgent(messages):
    print("reflect")
    res = Reflective_Chain.invoke(messages[-1].content)
    return [AIMessage(content=res.content)]

def CheckIteration(messages):
    print("check")
    if len(messages)>4:
        return "summarize"
    return "reflect"

def SummarizingAgent(messages):
    print("summarizing")
    user_input = messages[0].content
    therapist_response = messages[-1].content 
    res = Summarizer_Chain.invoke({"user_input":user_input,"response":therapist_response})
    return [AIMessage(content=res.content)]



graph.add_node("generate",GenerativeAgent)
graph.add_node("check",CheckIteration)
graph.add_node("reflect",ReflectiveAgent)
graph.add_node("summarize",SummarizingAgent)
graph.set_entry_point("generate")
graph.add_conditional_edges(
    "generate",
    CheckIteration,
    {"reflect":"reflect", "summarize":"summarize"}
    
)

#graph.add_edge("generate","check")
graph.add_edge("reflect","generate")
graph.add_edge("summarize",END)


MentalHealthAI = graph.compile()




