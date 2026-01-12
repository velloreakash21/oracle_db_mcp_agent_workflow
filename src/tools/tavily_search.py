"""
Tavily Search Tool for documentation retrieval.
Used by Doc Search Agent to find relevant documentation from the web.
"""
import os
from typing import Optional, List
from tavily import TavilyClient
from langchain_core.tools import tool

from src.config import settings

# Import tracer with graceful fallback
try:
    from src.telemetry import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    from contextlib import contextmanager
    class NoOpTracer:
        @contextmanager
        def start_as_current_span(self, name, **kwargs):
            class NoOpSpan:
                def set_attribute(self, k, v): pass
            yield NoOpSpan()
    tracer = NoOpTracer()


class TavilySearchTool:
    """Wrapper for Tavily search API."""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Tavily client with API key."""
        api_key = settings.tavily_api_key
        if api_key:
            try:
                self.client = TavilyClient(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Tavily client: {e}")

    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> dict:
        """
        Perform a web search using Tavily.

        Args:
            query: Search query
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            Search results dictionary
        """
        if not self.client:
            return {"error": "Tavily client not initialized. Check API key.", "results": []}

        with tracer.start_as_current_span("tavily_search") as span:
            span.set_attribute("search.query", query)
            span.set_attribute("search.depth", search_depth)
            span.set_attribute("search.max_results", max_results)

            try:
                results = self.client.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains
                )

                span.set_attribute("search.results_count", len(results.get("results", [])))

                return results

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                return {"error": str(e), "results": []}

    def get_search_context(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Get search results formatted for LLM context.

        Args:
            query: Search query
            max_tokens: Maximum context length

        Returns:
            Formatted search context string
        """
        if not self.client:
            return "Tavily client not initialized. Check API key."

        with tracer.start_as_current_span("tavily_get_context") as span:
            span.set_attribute("search.query", query)

            try:
                context = self.client.get_search_context(
                    query=query,
                    max_tokens=max_tokens
                )
                return context
            except Exception as e:
                span.set_attribute("error", True)
                return f"Search error: {str(e)}"


# Create singleton instance
_tavily_tool = TavilySearchTool()


@tool
def search_documentation(
    query: str,
    max_results: int = 5
) -> str:
    """
    Search the web for programming documentation and tutorials.

    Args:
        query: What to search for (e.g., "Python Oracle connection tutorial")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with titles, URLs, and content snippets
    """
    results = _tavily_tool.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_domains=[
            "docs.python.org",
            "oracle.com",
            "docs.oracle.com",
            "stackoverflow.com",
            "github.com",
            "realpython.com",
            "geeksforgeeks.org",
            "medium.com",
            "dev.to",
            "langchain.com",
            "python.langchain.com"
        ]
    )

    if "error" in results:
        return f"Search error: {results['error']}"

    if not results.get("results"):
        return "No results found for your query."

    # Format results
    formatted = []
    for i, result in enumerate(results["results"], 1):
        formatted.append(f"""
**Result {i}: {result.get('title', 'No title')}**
URL: {result.get('url', 'No URL')}
{result.get('content', 'No content')[:500]}...
""")

    return "\n---\n".join(formatted)


@tool
def search_oracle_docs(query: str) -> str:
    """
    Search specifically for Oracle database documentation.

    Args:
        query: Oracle-related search query

    Returns:
        Oracle documentation search results
    """
    full_query = f"Oracle database {query}"

    results = _tavily_tool.search(
        query=full_query,
        search_depth="advanced",
        max_results=5,
        include_domains=[
            "oracle.com",
            "docs.oracle.com",
            "blogs.oracle.com",
            "asktom.oracle.com"
        ]
    )

    if "error" in results:
        return f"Search error: {results['error']}"

    if not results.get("results"):
        return "No Oracle documentation found for your query."

    # Format results
    formatted = []
    for i, result in enumerate(results["results"], 1):
        formatted.append(f"""
**{i}. {result.get('title', 'No title')}**
{result.get('url', '')}
{result.get('content', '')[:400]}...
""")

    return "\n".join(formatted)


@tool
def search_python_docs(query: str) -> str:
    """
    Search specifically for Python documentation and tutorials.

    Args:
        query: Python-related search query

    Returns:
        Python documentation search results
    """
    full_query = f"Python {query}"

    results = _tavily_tool.search(
        query=full_query,
        search_depth="advanced",
        max_results=5,
        include_domains=[
            "docs.python.org",
            "realpython.com",
            "python.org",
            "pypi.org"
        ]
    )

    if "error" in results:
        return f"Search error: {results['error']}"

    if not results.get("results"):
        return "No Python documentation found for your query."

    # Format results
    formatted = []
    for i, result in enumerate(results["results"], 1):
        formatted.append(f"""
**{i}. {result.get('title', 'No title')}**
{result.get('url', '')}
{result.get('content', '')[:400]}...
""")

    return "\n".join(formatted)


@tool
def get_documentation_context(query: str) -> str:
    """
    Get comprehensive documentation context for a programming topic.
    Returns a larger context suitable for detailed explanations.

    Args:
        query: Topic to get documentation for

    Returns:
        Comprehensive documentation context
    """
    return _tavily_tool.get_search_context(
        query=query,
        max_tokens=4000
    )


# Export tools
__all__ = [
    "search_documentation",
    "search_oracle_docs",
    "search_python_docs",
    "get_documentation_context"
]


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test searches
    print("Testing Tavily Search...")

    result = search_documentation("Python Oracle database connection")
    print("\n=== Documentation Search ===")
    print(result[:1000])

    result = search_oracle_docs("connection pooling")
    print("\n=== Oracle Docs Search ===")
    print(result[:1000])
