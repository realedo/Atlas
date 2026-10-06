from ddgs import DDGS
from config import MX_RES, WEB_TOGGLE

def web_search(query: str, max_res: int = MX_RES) -> str:

    if not WEB_TOGGLE:
        return "Web search is disabled."

    query = query.strip()
    if not query:
        return "Web search requires a non-empty query."

    max_res = max(1, min(int(max_res), 10))
    sections = []

    try:
        ddgs = DDGS()

        # Regular web search
        try:
            results = list(
                ddgs.text(query, max_results=max_res) or []
            )

            if results:
                lines = ["## Web search results"]
                for i, result in enumerate(results, start=1):
                    lines.append(
                        f"{i}. Title: {result.get('title', 'Untitled')}\n"
                        f"   Snippet: {result.get('body', 'No snippet available')}\n"
                        f"   URL: {result.get('href', 'No URL available')}"
                    )
                sections.append("\n".join(lines))
        except Exception as e:
            sections.append(f"Web search failed: {e}")

        # News search, independent of regular web search
        try:
            news = list(
                ddgs.news(query, max_results=max_res) or []
            )

            if news:
                lines = ["## News search results"]
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

