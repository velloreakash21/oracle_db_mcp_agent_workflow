# Building Observable Multi-Agent AI Applications with LangGraph, Oracle Database, and OpenTelemetry

*A comprehensive guide to implementing distributed tracing in agentic AI systems*

---

## Introduction

As AI applications evolve from simple chatbots to complex multi-agent systems, understanding what happens inside them becomes increasingly challenging. When a user asks "How do I connect to Oracle database in Python?", multiple AI agents might spring into action - searching documentation, querying databases, and synthesizing responses. But when something goes wrong, or when you need to optimize performance, how do you know where to look?

This is where observability becomes crucial.

In this guide, we'll build a **Code Assistant** - an AI-powered tool that helps developers find documentation and code examples. More importantly, we'll implement comprehensive observability using **OpenTelemetry**, giving us full visibility into every agent interaction, LLM call, and database query.

### What You'll Learn

- Design a multi-agent system with LangGraph
- Connect to Oracle Database 23ai for storing code snippets
- Implement OpenTelemetry distributed tracing
- Visualize traces with Jaeger
- Debug and optimize AI applications

### The Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | Claude 3.5 Sonnet |
| Database | Oracle 23ai |
| Web Search | Tavily API |
| Observability | OpenTelemetry + Jaeger |
| Frontend | Streamlit |

Let's dive in!

---

## Architecture Overview

### The Big Picture

Our Code Assistant consists of three agents working together:

```
                    +-------------------+
                    |   User Query      |
                    |                   |
                    | "How do I use     |
                    | connection        |
                    | pooling?"         |
                    +---------+---------+
                              |
                              v
               +--------------------------+
               |    ORCHESTRATOR AGENT    |
               |                          |
               |  - Analyzes query        |
               |  - Routes to agents      |
               |  - Combines results      |
               +-------------+------------+
                             |
             +---------------+---------------+
             |                               |
             v                               v
     +---------------+               +---------------+
     |  DOC SEARCH   |               |  CODE QUERY   |
     |    AGENT      |               |    AGENT      |
     |               |               |               |
     | Tavily API    |               | Oracle SQLcl  |
     +---------------+               +---------------+
```

### Why This Architecture?

1. **Separation of Concerns**: Each agent specializes in one task
2. **Parallel Execution**: Agents can work simultaneously
3. **Flexibility**: Easy to add new agents
4. **Testability**: Each component can be tested independently

### The Observability Layer

Every interaction in this system generates telemetry. Here's what a typical trace looks like:

```
+------------------------------------------------------------+
|                    TRACE: user_query_123                    |
+------------------------------------------------------------+
|                                                            |
|  code_assistant_query ------------------------------------ |
|  |                                                        |
|  +-- orchestrator_analyze ------                          |
|  |                                                        |
|  +-- doc_search_agent --------------------                |
|  |   |                                                    |
|  |   +-- llm_invoke ----------                            |
|  |   |                                                    |
|  |   +-- tavily_search ------------                       |
|  |                                                        |
|  +-- code_query_agent --------------------                |
|  |   |                                                    |
|  |   +-- llm_invoke ----------                            |
|  |   |                                                    |
|  |   +-- oracle_query ---------                           |
|  |                                                        |
|  +-- orchestrator_combine ----------                      |
|                                                            |
+------------------------------------------------------------+
```

This trace tells us:
- Total request duration
- Time spent in each agent
- LLM latency
- Database query time
- Where bottlenecks occur

---

## Setting Up the Foundation

### Prerequisites

Before we begin, ensure you have:
- Python 3.11+
- Docker Desktop
- Anthropic API key
- Tavily API key (free tier: 1000 searches/month)

### Project Structure

```bash
mkdir code-assistant && cd code-assistant

# Create directory structure
mkdir -p src/{agents,tools,database,telemetry,frontend}
mkdir -p tests docs/diagrams

# Initialize Python packages
touch src/__init__.py
touch src/agents/__init__.py
touch src/tools/__init__.py
touch src/database/__init__.py
touch src/telemetry/__init__.py
```

### Docker Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  oracle-db:
    image: container-registry.oracle.com/database/free:latest
    container_name: oracle-23ai-code-assistant
    ports:
      - "1521:1521"
    environment:
      - ORACLE_PWD=CodeAssist123
    volumes:
      - oracle-data:/opt/oracle/oradata
    healthcheck:
      test: ["CMD", "sqlplus", "-L", "sys/CodeAssist123@//localhost:1521/FREE as sysdba", "@/dev/null"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger-code-assistant
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  oracle-data:
```

Start the services:

```bash
docker-compose up -d

# Wait for Oracle (first time takes ~5 minutes)
docker logs -f oracle-23ai-code-assistant
# Look for: "DATABASE IS READY TO USE!"
```

### Configuration Management

Create `src/config.py`:

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # LLM Configuration
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Tavily Search
    tavily_api_key: str = ""

    # Oracle Database
    oracle_host: str = "localhost"
    oracle_port: str = "1521"
    oracle_service: str = "FREEPDB1"
    oracle_user: str = "codeassist"
    oracle_password: str = "CodeAssist123"

    # OpenTelemetry
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "code-assistant"

    @property
    def oracle_dsn(self) -> str:
        return f"{self.oracle_host}:{self.oracle_port}/{self.oracle_service}"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### Database Schema

Our code snippets live in Oracle. Here's the schema:

```sql
CREATE TABLE code_snippets (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR2(200) NOT NULL,
    description VARCHAR2(2000),
    language VARCHAR2(50) NOT NULL,
    framework VARCHAR2(100),
    category VARCHAR2(100),
    difficulty VARCHAR2(20) DEFAULT 'intermediate',
    code CLOB NOT NULL,
    tags VARCHAR2(500),
    source_url VARCHAR2(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for query performance
CREATE INDEX idx_snippets_language ON code_snippets(language);
CREATE INDEX idx_snippets_category ON code_snippets(category);
CREATE INDEX idx_snippets_framework ON code_snippets(framework);
```

---

## Building the Agents

### Understanding LangGraph

LangGraph extends LangChain with graph-based workflows. Instead of linear chains, we define **nodes** (processing steps) and **edges** (transitions).

Key concepts:
- **State**: Shared data passed between nodes
- **Nodes**: Functions that process state
- **Edges**: Connections between nodes (can be conditional)

### The Code Query Agent

This agent searches our Oracle database for code snippets:

```python
"""
Code Query Agent - Searches Oracle database for code snippets.
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.tools.oracle_mcp import search_code_snippets, get_snippet_by_id
from src.config import settings
from src.telemetry import get_tracer

tracer = get_tracer(__name__)

class CodeQueryState(TypedDict):
    """State passed through the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    results: str

SYSTEM_PROMPT = """You are a Code Query Agent specialized in finding
code snippets from a database. Use your tools to search for relevant examples."""

def create_code_query_agent():
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0
    )

    tools = [search_code_snippets, get_snippet_by_id]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: CodeQueryState) -> CodeQueryState:
        """Main reasoning node with tracing."""
        with tracer.start_as_current_span("code_query_reasoning") as span:
            span.set_attribute("agent.name", "code_query")

            messages = state["messages"]
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

            response = llm_with_tools.invoke(messages)
            span.set_attribute("has_tool_calls", bool(response.tool_calls))

            return {"messages": [response]}

    def should_continue(state: CodeQueryState) -> str:
        """Determine next step based on LLM response."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    workflow = StateGraph(CodeQueryState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
```

**Key Points:**
1. We wrap the agent logic in a span for tracing
2. The `should_continue` function creates a loop between agent and tools
3. Tool results feed back into the agent for reasoning

### The Oracle Query Tool

```python
@tool
def search_code_snippets(
    language: str = None,
    category: str = None,
    keyword: str = None,
    limit: int = 5
) -> str:
    """
    Search for code snippets in the Oracle database.

    Args:
        language: Filter by programming language (python, java, sql)
        category: Filter by category (database, api, ai, auth)
        keyword: Search in title, description, and tags
        limit: Maximum results to return (default: 5)
    """
    with tracer.start_as_current_span("oracle_query") as span:
        span.set_attribute("db.system", "oracle")

        # Build query with sanitized inputs
        conditions = []
        params = {}

        if language:
            conditions.append("LOWER(language) = LOWER(:language)")
            params["language"] = language
        if category:
            conditions.append("LOWER(category) = LOWER(:category)")
            params["category"] = category
        if keyword:
            conditions.append("""(
                LOWER(title) LIKE LOWER(:keyword)
                OR LOWER(tags) LIKE LOWER(:keyword)
            )""")
            params["keyword"] = f"%{keyword}%"

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT id, title, description, language, framework,
                   category, code, tags
            FROM code_snippets
            WHERE {where_clause}
            ORDER BY created_at DESC
            FETCH FIRST {min(limit, 20)} ROWS ONLY
        """

        span.set_attribute("db.statement", query[:500])

        # Execute query
        result = execute_oracle_query(query, params)
        span.set_attribute("db.rows_returned", len(result))

        return json.dumps(result, indent=2, default=str)
```

### The Doc Search Agent

This agent uses Tavily to search the web for documentation:

```python
"""
Doc Search Agent - Searches the web for programming documentation.
"""
from src.tools.tavily_search import (
    search_documentation,
    search_oracle_docs,
    search_python_docs
)

DOC_SEARCH_SYSTEM_PROMPT = """You are a Documentation Search Agent
specialized in finding programming documentation and tutorials.

Guidelines:
- For Oracle-specific queries, use search_oracle_docs
- For Python-specific queries, use search_python_docs
- For general queries, use search_documentation
- Summarize key points from search results
- Include relevant URLs for further reading
"""

def create_doc_search_agent():
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0
    )

    tools = [
        search_documentation,
        search_oracle_docs,
        search_python_docs
    ]

    llm_with_tools = llm.bind_tools(tools)

    # Similar graph structure to Code Query Agent...
```

### The Orchestrator Agent

The orchestrator coordinates between agents:

```python
"""
Orchestrator Agent - Coordinates between Doc Search and Code Query agents.
"""
def create_orchestrator_agent():
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
                "how", "what", "why", "explain", "documentation"
            ])

            needs_code = any(word in query for word in [
                "code", "example", "snippet", "implement", "show me"
            ])

            # Default: call both if unclear
            if not needs_docs and not needs_code:
                needs_docs = True
                needs_code = True

            if needs_docs:
                agents_to_call.append("doc_search")
            if needs_code:
                agents_to_call.append("code_query")

            span.set_attribute("agents_to_call", str(agents_to_call))
            return {"agents_to_call": agents_to_call}

    def combine_results(state: OrchestratorState) -> OrchestratorState:
        """Combine results from both agents into final response."""
        with tracer.start_as_current_span("orchestrator_combine"):
            doc_results = state.get("doc_results", "")
            code_results = state.get("code_results", "")

            synthesis_prompt = f"""Synthesize this information:

            Documentation: {doc_results}
            Code Examples: {code_results}

            Provide a clear, helpful response."""

            response = llm.invoke([
                SystemMessage(content="Synthesize information for developers."),
                HumanMessage(content=synthesis_prompt)
            ])

            return {"final_response": response.content}
```

---

## Implementing Observability

### OpenTelemetry Basics

OpenTelemetry provides three pillars of observability:
1. **Traces**: Request flow through distributed systems
2. **Metrics**: Quantitative measurements
3. **Logs**: Discrete events

For AI applications, traces are most valuable - they show us the journey of a request through multiple agents.

### Setting Up Tracing

```python
"""
OpenTelemetry configuration for Code Assistant.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

_telemetry_initialized = False

def init_telemetry(service_name: str = "code-assistant"):
    """Initialize OpenTelemetry with OTLP export to Jaeger."""
    global _telemetry_initialized

    if _telemetry_initialized:
        return

    # Create resource identifying our service
    resource = Resource.create({SERVICE_NAME: service_name})

    # Create and configure tracer provider
    provider = TracerProvider(resource=resource)

    # Export to Jaeger via OTLP
    otlp_exporter = OTLPSpanExporter(
        endpoint="localhost:4317",
        insecure=True
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global provider
    trace.set_tracer_provider(provider)

    _telemetry_initialized = True
    print(f"Telemetry initialized: {service_name}")

def get_tracer(name: str):
    """Get tracer with graceful fallback."""
    try:
        return trace.get_tracer(name)
    except Exception:
        return NoOpTracer()
```

### Graceful Degradation

The application works even if telemetry fails to initialize:

```python
class NoOpSpan:
    """No-op span when telemetry is unavailable."""
    def set_attribute(self, key, value): pass
    def add_event(self, name, attributes=None): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

class NoOpTracer:
    """No-op tracer when telemetry is unavailable."""
    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        yield NoOpSpan()
```

### Tracing Decorators

For cleaner code, use a decorator:

```python
def traced(name: str = None):
    """Decorator for tracing functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@traced("llm_invoke")
def call_llm(messages):
    return llm.invoke(messages)
```

---

## Visualizing Traces with Jaeger

### Accessing the UI

Open http://localhost:16686 in your browser.

### Finding Traces

1. Select service: `code-assistant`
2. Set time range (last hour)
3. Click "Find Traces"

### Reading a Trace

A typical trace shows:

| Span | Duration | Purpose |
|------|----------|---------|
| code_assistant_query | 2.3s | Total request |
| orchestrator_analyze | 45ms | Route decision |
| doc_search_agent | 1.1s | Web search |
| llm_invoke | 650ms | LLM reasoning |
| tavily_search | 420ms | API call |
| code_query_agent | 890ms | DB search |
| llm_invoke | 580ms | LLM reasoning |
| oracle_query | 95ms | Database |
| orchestrator_combine | 290ms | Synthesis |

### What to Look For

1. **Total Duration**: Is the request fast enough?
2. **LLM Latency**: Typically the longest spans
3. **Database Time**: Should be under 100ms
4. **Parallel vs Sequential**: Are agents running in parallel?

---

## Best Practices

### Span Naming Conventions

Use consistent, hierarchical names:
- `service.operation` format
- Examples: `orchestrator.analyze`, `doc_search.invoke`

### Essential Attributes

| Category | Attributes |
|----------|------------|
| Query | query, query.length |
| LLM | llm.model, llm.duration_ms, llm.has_tool_calls |
| Database | db.system, db.statement, db.rows_returned |
| Error | error, error.message |

### Error Handling

Always record errors in spans:

```python
from opentelemetry.trace import Status, StatusCode

try:
    result = risky_operation()
    span.set_status(Status(StatusCode.OK))
except Exception as e:
    span.set_status(Status(StatusCode.ERROR, str(e)))
    span.record_exception(e)
    raise
```

### Input Validation

Always validate and sanitize inputs to prevent injection:

```python
def sanitize_input(value: str, max_length: int = 100) -> str:
    """Sanitize input to prevent SQL injection."""
    if not value:
        return ""
    sanitized = value.replace("'", "''").replace(";", "").replace("--", "")
    return sanitized[:max_length]
```

---

## Troubleshooting Guide

### No Traces Appearing

1. **Check Jaeger is running**: `docker ps | grep jaeger`
2. **Verify OTLP endpoint**: Should be `localhost:4317`
3. **Ensure `init_telemetry()` is called** before any operations
4. **Check for exceptions** in telemetry initialization

### Incomplete Traces

- Call `shutdown_telemetry()` before exit
- BatchSpanProcessor needs time to flush (a few seconds)
- Increase `max_export_batch_size` if needed

### High Latency

- Check LLM response times (typically the bottleneck)
- Verify database indexes exist
- Consider caching for repeated queries
- Use connection pooling for database

### Oracle Connection Issues

```bash
# Check Oracle is running
docker logs oracle-23ai-code-assistant | tail -20

# Test connection
python -c "import oracledb; print(oracledb.connect(user='codeassist', password='CodeAssist123', dsn='localhost:1521/FREEPDB1').version)"
```

---

## Conclusion

Building observable AI applications isn't optional - it's essential. As your agents become more complex, the ability to see inside them becomes invaluable.

In this guide, we've:
- Built a multi-agent system with LangGraph
- Connected to Oracle Database 23ai
- Implemented comprehensive tracing with OpenTelemetry
- Visualized everything in Jaeger

The patterns shown here scale from simple chatbots to complex enterprise AI systems.

### Key Takeaways

1. **Trace everything**: LLM calls, database queries, tool invocations
2. **Use meaningful span names**: Make traces readable
3. **Add relevant attributes**: Enable filtering and analysis
4. **Implement graceful degradation**: App works even if telemetry fails

### Next Steps

- Add metrics for monitoring (request rates, error rates)
- Implement alerting on slow traces
- Build a trace analysis dashboard
- Add semantic search with Oracle AI Vector Search

### Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Oracle 23ai Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/)
- [Jaeger Tracing](https://www.jaegertracing.io/)
- [Tavily API](https://tavily.com/)

---

*Full source code available in the [repository](https://github.com/velloreakash21/oracle_db_mcp_agent_workflow).*

---

**Author: [Vellore Akash](https://www.linkedin.com/in/velloreakash/)** - AI/ML Architect with experience building enterprise AI platforms using LangChain, LlamaIndex, and cloud-native technologies.
