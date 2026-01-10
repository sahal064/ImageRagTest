from langchain.tools import tool
from retrieval.search import search_images

@tool
def image_search_tool(query:str) -> dict:
    """Search images from vector DB"""
    results=search_images(query,k=5)
    return results[0] if results else {}