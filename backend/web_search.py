import re
import trafilatura
from ddgs import DDGS
from config import MX_RES, WEB_TOGGLE, append_links, append_images, max_url_scrape_len


url_pattern = re.compile(r'(https?://[^\s]+)')


def search_url(url: str, max_len: int =max_url_scrape_len) -> str:
    
    try:
        
        downloaded_data = trafilatura.fetch_url(url)
        if downloaded_data:
            
            text = trafilatura.extract(downloaded_data, include_links=append_links, include_images=append_images)
            if text:
                print(f"extracted context from given URL({url})\n")
                return text[:max_len] + f"...\n Articole has been cutted due to being longer than max len ({max_len})"
    
    except Exception as e:
        print(f"An error has occured qhile fetchijg data from {url}: {str(e)}")
        return "Could not fetch data from provided url"
            






def web_search(query: str, max_res: int = MX_RES) -> str:

    if not WEB_TOGGLE:
        return "Web search is disabled."

    query = query.strip()
    if not query:
        return "Web search requires a non-empty query."

    #max_res = max(1, min(int(max_res), 10))
    sections = []
    
    #cheks if theres any urls in the prompt
    urls_present = url_pattern.findall(query)
    if urls_present:
        url_to_search = urls_present[0]
        sections.append(f"Content from the searched URL ({url_to_search}): \n" + search_url(url_to_search))
        return f"Searched the URL given in the prompt: {query}\n\n" + "\n\n".join(sections)
    
    #if no direct links try this instead
    try:
        ddgs = DDGS()


        try:
            #searches the actual prompt here(ddg search). might need to inference the prompt to give out a better query, seems costly in execute time but might give out better context.
            results = list(
                ddgs.text(query, max_results=max_res) or []
            )

            if results:
                lines = ["Web search results"]
                for i, result in enumerate(results, start=1):
                    lines.append(
                        f"{i}. Title: {result.get('title', 'Untitled')}\n"
                        f"   Snippet: {result.get('body', 'No snippet available')}\n"
                        f"   URL: {result.get('href', 'No URL available')}"
                    )
                    
                #use trafilatura.extract on the first page hitted
                fist_url = results[0].get('href')
                if fist_url:
                    lines.append("\nData from the first page hitted: \n") 
                    print(f"attempting to extarct data from first result\n")
                    lines.append(search_url(fist_url))
                    
                #join ddgs and context from trafialtura on the first href
                sections.append("\n".join(lines))
        
        except Exception as e:
            sections.append(f"Web search failed: {e}")

        
        try:
            #same as for normal ddg search 
            news = list(
                ddgs.news(query, max_results=max_res) or []
            )

            if news:
                lines = ["News search results"]
                for i, result in enumerate(news, start=1):
                    lines.append(
                        f"{i}. Title: {result.get('title', 'Untitled')}\n"
                        f"   Snippet: {result.get('body', 'No snippet available')}\n"
                        f"   URL: {result.get('url', result.get('href', 'No URL available'))}\n"
                        f"   Date: {result.get('date', 'Unknown')}"
                    )
                sections.append("\n".join(lines))

        except Exception as e:
            sections.append(f"News search failed: {e}")

    except Exception as e:
        return f"Web search initialization failed: {e}"

    if not sections:
        return "No web or news results found for this query."

    return f"Search query: {query}\n\n" + "\n\n".join(sections)

