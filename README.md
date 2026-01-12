# Oracle Database MCP Agent Workflow

A multi-agent Code Assistant built with LangGraph, Oracle Database 23ai, and OpenTelemetry for distributed tracing.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)
![Oracle](https://img.shields.io/badge/Oracle-23ai-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

## About

This project showcases a production-ready multi-agent orchestration system that helps developers find documentation and code examples. The architecture leverages my experience building enterprise AI platforms with LangChain, LlamaIndex, and cloud-native technologies.

**Key highlights:**
- Intelligent query routing between specialized agents
- Oracle Database 23ai integration for code snippet retrieval
- Real-time documentation search via Tavily API
- Full observability with OpenTelemetry and Jaeger
- Clean Streamlit interface for demonstration

## Architecture

```
                        ┌─────────────────┐
                        │   User Query    │
                        └────────┬────────┘
                                 │
                                 ▼
                   ┌─────────────────────────┐
                   │   Orchestrator Agent    │
                   │  (Query Analysis &      │
                   │   Routing Logic)        │
                   └───────────┬─────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   Doc Search Agent      │       │   Code Query Agent      │
│   ─────────────────     │       │   ─────────────────     │
│   • Tavily API          │       │   • Oracle 23ai         │
│   • Web documentation   │       │   • 40+ code snippets   │
│   • Real-time results   │       │   • Full-text search    │
└───────────┬─────────────┘       └───────────┬─────────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Combined Response  │
                  │  + Trace Metadata   │
                  └─────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Framework | LangGraph |
| LLM | Claude 3.5 Sonnet |
| Database | Oracle Database 23ai (Free) |
| Web Search | Tavily API |
| Observability | OpenTelemetry + Jaeger |
| Frontend | Streamlit |
| Containerization | Docker Compose |

## Prerequisites

- Python 3.11+
- Docker Desktop
- [Anthropic API Key](https://console.anthropic.com/)
- [Tavily API Key](https://tavily.com/)

## Setup

### 1. Clone and Configure

```bash
git clone https://github.com/velloreakash21/oracle_db_mcp_agent_workflow.git
cd oracle_db_mcp_agent_workflow

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

### 2. Start Services

```bash
docker-compose up -d
```

Wait for Oracle to be healthy (~2-3 minutes):
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### 3. Initialize Database

```bash
# Create Oracle user
docker exec -i oracle-23ai-code-assistant sqlplus sys/CodeAssist123@FREEPDB1 as sysdba <<'EOF'
CREATE USER codeassist IDENTIFIED BY CodeAssist123;
GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE TRIGGER, CREATE SEQUENCE, UNLIMITED TABLESPACE TO codeassist;
EXIT;
EOF

# Create schema
docker exec -i oracle-23ai-code-assistant sqlplus codeassist/CodeAssist123@FREEPDB1 <<'EOF'
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
    author VARCHAR2(100) DEFAULT 'Code Assistant',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EXIT;
EOF

# Seed data
python -m src.database.seed_data
```

### 4. Run

```bash
streamlit run streamlit_app.py
```

**Access:**
- Application: http://localhost:8501
- Jaeger Traces: http://localhost:16686

## Usage

### Streamlit UI

Ask questions like:
- "How do I connect to Oracle database in Python?"
- "Show me connection pooling examples"
- "How do I create a FastAPI endpoint?"
- "Show me LangChain agent examples"

### CLI

```bash
python -m src.main "How do I handle Oracle transactions?"
```

### Python API

```python
from src.agents.orchestrator import ask_assistant

response = ask_assistant("Show me JWT authentication examples")
print(response)
```

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── orchestrator.py     # Query routing and response aggregation
│   │   ├── doc_search.py       # Tavily documentation search
│   │   └── code_query.py       # Oracle database queries
│   ├── tools/
│   │   ├── tavily_search.py    # Tavily API wrapper
│   │   └── oracle_mcp.py       # Oracle connection utilities
│   ├── database/
│   │   ├── schema.sql          # Table definitions
│   │   └── seed_data.py        # 40+ code snippets
│   ├── telemetry/
│   │   └── tracing.py          # OpenTelemetry configuration
│   └── frontend/
│       ├── app.py              # Streamlit application
│       ├── components.py       # UI components
│       └── styles.py           # Custom styling
├── docker-compose.yml          # Oracle 23ai + Jaeger
├── requirements.txt
└── streamlit_app.py            # Entry point
```

## Code Snippets Database

| Category | Count | Topics |
|----------|-------|--------|
| database | 12 | Oracle connections, queries, transactions, pooling |
| api | 8 | FastAPI endpoints, middleware, error handling |
| ai | 8 | LangChain, LangGraph, Tavily, embeddings |
| auth | 5 | JWT, OAuth, RBAC patterns |
| data | 4 | Pandas, CSV, JSON processing |
| testing | 3 | Pytest fixtures, mocking |

## Observability

Traces are captured for each query:
- `orchestrator_analyze` - Initial query analysis
- `doc_search_agent` - Documentation retrieval
- `code_query_agent` - Database lookup
- `orchestrator_combine` - Response generation

View detailed spans and timing at http://localhost:16686

## Configuration

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `TAVILY_API_KEY` | Tavily search key |
| `ORACLE_HOST` | Database host (default: localhost) |
| `ORACLE_PORT` | Database port (default: 1521) |
| `ORACLE_SERVICE` | Service name (default: FREEPDB1) |

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Vellore Akash**
AI/ML Architect

- [LinkedIn](https://www.linkedin.com/in/velloreakash/)
- [GitHub](https://github.com/velloreakash21)
