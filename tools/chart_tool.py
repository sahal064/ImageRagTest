from langchain.tools import tool

@tool
def chart_reasoning_tool(image_metadata:dict,question:str)->dict:
    "Provides The Structured Chart Data for LLM Reasoning"

    return {
        "data":image_metadata.get("data"),
        "inference": image_metadata.get("inference"),
        "question":question
    }