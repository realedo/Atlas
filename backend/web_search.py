from ddgs import DDGS
from config import MX_RES

def web_search(query: str, max_res=MX_RES) -> str:
    
    try:
        results = DDGS().text(query, max_results=max_res)
        if not results:
            return "No results found on the web. "
        
        context = "Web Search results:\n"
        
        for r in results:
           
            context += f"Title: {r.get('title')}\n Snippet: {r.get('body')}\n URL: {r.get('href')}\n\n"
            print(f"context: {context}\n")
            return context
    except Exception as e:
        return f"An error occured during the web search: {str(e)}\n"

