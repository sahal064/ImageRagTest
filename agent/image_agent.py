from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from config.settings import GROQ_API_KEY,GROQ_MODEL,TEMPERATURE
from tools.image_search_tool import image_search_tool
from tools.chart_tool import chart_reasoning_tool

def create_image_agent() -> Runnable:
    """
    Creates a Groq-powered image understanding agent
    with native tool calling
    """

    llm=ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=TEMPERATURE
    )

    prompt=ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=""" 
You are an intelligent image-understanding agent.

You can:
- Search images using a vector database
- Read structured chart data from metadata

Rules:
- If the user asks for visual description → call image_search_tool
- If the user asks analytical / numerical / trend-based questions →
  1. Call image_search_tool
  2. Use chart data to answer
- NEVER hallucinate values
- Use ONLY provided metadata
- If data is missing, say you don't have enough information
                          """),
                          ("human","{input}")
        ]
    )

    tools=[
        image_search_tool,chart_reasoning_tool
    ]

    ll_with_tools=llm.bind_tools(tools)

    agent= prompt | ll_with_tools

    return agent