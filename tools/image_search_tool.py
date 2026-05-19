from langchain.tools import tool
from retrieval.search import search_images
from typing import List


@tool
def image_search_tool(query: str, top_k: int = 5) -> List[dict]:
    """Search images from vector DB"""
    query = query.strip()
    if not query:
        return []
    return search_images(query, k=top_k)
