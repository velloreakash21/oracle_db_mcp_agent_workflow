"""
Orchestrator Agent - Coordinates between Doc Search and Code Query agents.
Routes queries and combines results for comprehensive answers.
"""
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.agents.doc_search import search_docs
from src.agents.code_query import query_code_snippets
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


# Orchestrator State
class OrchestratorState(TypedDict):
    """State for the Orchestrator Agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    doc_results: str
    code_results: str
    final_response: str
    agents_to_call: list[str]


# System prompt for orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are an Orchestrator Agent that helps developers find information and code examples.

You coordinate two specialized agents:
1. **Doc Search Agent**: Searches web for documentation, tutorials, and explanations
2. **Code Query Agent**: Searches a database of code snippets and examples

Your job:
1. Analyze the user's query
2. Decide which agent(s) to use:
   - Use Doc Search for: explanations, concepts, "how does X work?", best practices
   - Use Code Query for: code examples, implementations, "show me code for X"
   - Use BOTH for: comprehensive help like "How do I implement X?" (needs both explanation AND code)
3. Combine results into a helpful, coherent response

Response format:
- Start with a brief explanation (from docs if available)
- Include relevant code examples (from code query if available)
- Add tips or best practices
- Keep it practical and actionable

Be helpful and concise. Developers want answers, not fluff.
"""


def create_orchestrator_agent():
    """Create and return the Orchestrator Agent graph."""

    # Initialize LLM
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0
    )

    def analyze_query(state: OrchestratorState) -> OrchestratorState:
        """Analyze query and decide which agents to call."""
        with tracer.start_as_current_span("orchestrator_analyze") as span:
            query = state["query"].lower()
            span.set_attribute("query", query)

            agents_to_call = []

            # Heuristics for routing
            needs_docs = any(word in query for word in [
                "how", "what", "why", "explain", "concept", "best practice",
                "documentation", "tutorial", "guide", "learn"
            ])

            needs_code = any(word in query for word in [
                "code", "example", "snippet", "implement", "show me",
                "sample", "function", "class", "script"
            ])

            # Default: if unclear, call both
            if not needs_docs and not needs_code:
                needs_docs = True
                needs_code = True

            if needs_docs:
                agents_to_call.append("doc_search")
            if needs_code:
                agents_to_call.append("code_query")

            span.set_attribute("agents_to_call", str(agents_to_call))

            return {"agents_to_call": agents_to_call}

    def call_doc_search(state: OrchestratorState) -> OrchestratorState:
        """Call the Doc Search Agent."""
        with tracer.start_as_current_span("orchestrator_call_doc_search") as span:
            query = state["query"]
            span.set_attribute("query", query)

            result = search_docs(query)

            span.set_attribute("result_length", len(result))

            return {"doc_results": result}

    def call_code_query(state: OrchestratorState) -> OrchestratorState:
        """Call the Code Query Agent."""
        with tracer.start_as_current_span("orchestrator_call_code_query") as span:
            query = state["query"]
            span.set_attribute("query", query)

            result = query_code_snippets(query)

            span.set_attribute("result_length", len(result))

            return {"code_results": result}

    def combine_results(state: OrchestratorState) -> OrchestratorState:
        """Combine results from both agents into final response."""
        with tracer.start_as_current_span("orchestrator_combine") as span:
            doc_results = state.get("doc_results", "")
            code_results = state.get("code_results", "")
            query = state["query"]

            # Build prompt for final synthesis
            synthesis_prompt = f"""Based on the user's question and the gathered information, provide a comprehensive answer.

**User Question:** {query}

**Documentation/Explanation:**
{doc_results if doc_results else "No documentation found."}

**Code Examples:**
{code_results if code_results else "No code examples found."}

**Your Task:**
Synthesize this information into a clear, helpful response that:
1. Explains the concept briefly (if docs available)
2. Shows relevant code examples (if code available)
3. Provides practical tips
4. Is well-formatted with headers and code blocks

Keep it concise but complete."""

            messages = [
                SystemMessage(content="You are a helpful coding assistant synthesizing information for developers."),
                HumanMessage(content=synthesis_prompt)
            ]

            response = llm.invoke(messages)

            span.set_attribute("response_length", len(response.content))

            return {
                "final_response": response.content,
                "messages": [AIMessage(content=response.content)]
            }

    def route_to_agents(state: OrchestratorState) -> list[str]:
        """Route to the appropriate agents."""
        agents = state.get("agents_to_call", [])

        if not agents:
            return ["combine"]

        return agents

    # Build the graph
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("doc_search", call_doc_search)
    workflow.add_node("code_query", call_code_query)
    workflow.add_node("combine", combine_results)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Route based on analysis
    workflow.add_conditional_edges(
        "analyze",
        lambda s: route_to_agents(s),
        {
            "doc_search": "doc_search",
            "code_query": "code_query",
            "combine": "combine"
        }
    )

    # After doc search, check if we also need code
    workflow.add_conditional_edges(
        "doc_search",
        lambda s: "code_query" if "code_query" in s.get("agents_to_call", []) else "combine",
        {
            "code_query": "code_query",
            "combine": "combine"
        }
    )

    # Code query always goes to combine
    workflow.add_edge("code_query", "combine")

    # Combine ends the workflow
    workflow.add_edge("combine", END)

    return workflow.compile()


def ask_assistant(query: str) -> str:
    """
    Main entry point for the Code Assistant.

    Args:
        query: User's question about coding

    Returns:
        Comprehensive response with docs and code examples
    """
    with tracer.start_as_current_span("code_assistant_query") as span:
        span.set_attribute("query", query)

        orchestrator = create_orchestrator_agent()

        result = orchestrator.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "doc_results": "",
            "code_results": "",
            "final_response": "",
            "agents_to_call": []
        })

        response = result.get("final_response", "Sorry, I couldn't find an answer.")

        span.set_attribute("response_length", len(response))

        return response


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize telemetry for testing
    try:
        from src.telemetry.tracing import init_telemetry
        init_telemetry()
    except Exception:
        pass

    # Test queries
    test_queries = [
        "How do I connect to Oracle database in Python?",
        "Show me FastAPI authentication examples",
        "What is connection pooling and how do I implement it with Oracle?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        result = ask_assistant(query)
        print(result)
