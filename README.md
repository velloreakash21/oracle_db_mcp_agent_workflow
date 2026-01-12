# Oracle Database MCP Agent Workflow

An AI-powered Code Assistant that leverages multi-agent orchestration to help developers find documentation and code examples. Built with LangGraph, Oracle Database 23ai, and OpenTelemetry.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)
![Oracle](https://img.shields.io/badge/Oracle-23ai-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project demonstrates a sophisticated multi-agent system that combines:
- **LangGraph** for agent orchestration
- **Oracle Database 23ai** for code snippet storage and retrieval
- **Tavily API** for real-time documentation search
- **OpenTelemetry + Jaeger** for distributed tracing
- **Streamlit** for an interactive demo UI

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                         │
│              (Analyzes & Routes Queries)                     │
└─────────────────────────────────────────────────────────────┘
                   │                    │
                   ▼                    ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    Doc Search Agent      │  │    Code Query Agent      │
│      (Tavily API)        │  │     (Oracle 23ai)        │
│                          │  │                          │
│  • Web documentation     │  │  • Code snippets         │
│  • Real-time search      │  │  • 40+ examples          │
│  • Multiple sources      │  │  • Categorized by topic  │
└──────────────────────────┘  └──────────────────────────┘
                   │                    │
                   └────────┬───────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Combined Response                          │
│         (Documentation + Code Examples + Sources)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                             │
│    • Chat Interface    • Agent Activity    • Trace Viz      │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Orchestration**: Intelligent routing of queries to specialized agents
- **Oracle Database Integration**: Code snippets stored in Oracle 23ai with full-text search
- **Real-time Documentation**: Tavily API integration for up-to-date documentation
- **Distributed Tracing**: Full observability with OpenTelemetry and Jaeger
- **Interactive UI**: Professional Streamlit interface with trace visualization
- **Extensible Design**: Easy to add new agents and capabilities

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | Claude 3.5 Sonnet |
| Database | Oracle Database 23ai (Free) |
| Web Search | Tavily API |
| Tracing | OpenTelemetry + Jaeger |
| Frontend | Streamlit |
| Container | Docker |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- API Keys:
  - [Anthropic API Key](https://console.anthropic.com/)
  - [Tavily API Key](https://tavily.com/) (Free tier: 1000 searches/month)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/velloreakash21/oracle_db_mcp_agent_workflow.git
   cd oracle_db_mcp_agent_workflow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start Docker services**
   ```bash
   docker-compose up -d
   ```

6. **Initialize database** (wait for Oracle to be healthy first)
   ```bash
   # Check Oracle health
   docker ps --format "table {{.Names}}\t{{.Status}}"

   # Create user and schema
   docker exec -i oracle-23ai-code-assistant sqlplus sys/CodeAssist123@FREEPDB1 as sysdba <<'EOF'
   CREATE USER codeassist IDENTIFIED BY CodeAssist123;
   GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE TRIGGER, CREATE SEQUENCE, UNLIMITED TABLESPACE TO codeassist;
   EXIT;
   EOF

   # Create tables
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

7. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

8. **Open in browser**
   - **Code Assistant**: http://localhost:8501
   - **Jaeger UI**: http://localhost:16686

## Usage

### Example Queries

**Database Connections:**
```
How do I connect to Oracle database in Python?
Show me connection pooling examples with Oracle
```

**API Development:**
```
How do I create a REST API with FastAPI?
Show me JWT authentication examples
```

**AI/LLM Integration:**
```
How do I create a LangChain agent?
Show me how to use Tavily for web search
```

### CLI Mode

```bash
# Single query
python -m src.main "How do I connect to Oracle database?"

# Interactive mode
python -m src.main
```

### Python API

```python
from src.agents import ask_assistant

response = ask_assistant("How do I use connection pooling with Oracle?")
print(response)
```

## Project Structure

```
oracle_db_mcp_agent_workflow/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py    # Main coordinator agent
│   │   ├── doc_search.py      # Documentation search agent
│   │   └── code_query.py      # Database query agent
│   ├── tools/
│   │   ├── tavily_search.py   # Tavily API wrapper
│   │   └── oracle_mcp.py      # Oracle database tools
│   ├── database/
│   │   ├── schema.sql         # Database schema
│   │   ├── init_schema.py     # Schema initialization
│   │   └── seed_data.py       # Sample code snippets
│   ├── telemetry/
│   │   └── tracing.py         # OpenTelemetry setup
│   ├── frontend/
│   │   ├── app.py             # Streamlit main app
│   │   ├── components.py      # UI components
│   │   └── styles.py          # Custom CSS
│   ├── config.py              # Configuration
│   └── main.py                # CLI entry point
├── docs/
│   └── blog_post.md           # Technical blog post
├── docker-compose.yml         # Oracle + Jaeger containers
├── requirements.txt           # Python dependencies
├── demo.md                    # Demo guide
├── streamlit_app.py           # Streamlit entry point
└── .env.example               # Environment template
```

## Code Snippets Database

The database contains 40 pre-loaded code snippets:

| Category | Count | Topics |
|----------|-------|--------|
| database | 12 | Oracle connections, queries, transactions, pooling |
| api | 8 | FastAPI endpoints, middleware, error handling |
| ai | 8 | LangChain, LangGraph, Tavily, embeddings |
| auth | 5 | JWT, OAuth, RBAC patterns |
| data | 4 | Pandas, CSV, JSON processing |
| testing | 3 | Pytest fixtures, mocking |

## Observability

### Jaeger Tracing

View detailed traces at http://localhost:16686

Each query generates spans for:
- `orchestrator_analyze` - Query analysis
- `doc_search_agent` - Tavily documentation search
- `code_query_agent` - Oracle database query
- `orchestrator_combine` - Response generation

### Metrics

The UI displays real-time metrics:
- Total response time
- LLM inference time
- Database query time
- Web search time

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |
| `ORACLE_HOST` | Oracle host (default: localhost) | No |
| `ORACLE_PORT` | Oracle port (default: 1521) | No |
| `ORACLE_SERVICE` | Service name (default: FREEPDB1) | No |
| `ORACLE_USER` | Database user (default: codeassist) | No |
| `ORACLE_PASSWORD` | Database password | No |

## Development

### Adding New Agents

1. Create agent in `src/agents/`
2. Define state and tools
3. Register in orchestrator
4. Add tracing spans

### Adding Code Snippets

```python
# In src/database/seed_data.py
snippets.append({
    "title": "Your Snippet Title",
    "description": "Description",
    "language": "python",
    "category": "database",
    "code": """your code here""",
    "tags": "tag1,tag2"
})
```

## Troubleshooting

### Oracle Connection Issues
```bash
# Check Oracle status
docker logs oracle-23ai-code-assistant

# Restart Oracle
docker-compose restart oracle-db
```

### Streamlit Issues
```bash
# Clear cache
find . -name "__pycache__" -type d -exec rm -rf {} +
streamlit run streamlit_app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [Oracle Database](https://www.oracle.com/database/) - Enterprise database
- [Tavily](https://tavily.com/) - AI-powered search
- [Anthropic](https://www.anthropic.com/) - Claude LLM
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework

---

**Built for Oracle Developer Evangelist Technical Assignment**
