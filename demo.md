# Code Assistant Demo Guide

> Complete guide for recording the demo video of the AI-powered Code Assistant

---

## Quick Start Commands

### 1. Start Docker Services

```bash
cd /Users/velloreakash/claude/oracle_task/implementation/code-assistant
docker-compose up -d
```

**Wait for Oracle to be healthy (~2-3 minutes on first run):**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected output:
```
NAMES                        STATUS
oracle-23ai-code-assistant   Up X minutes (healthy)
jaeger-code-assistant        Up X minutes
```

### 2. Initialize Database (First Time Only)

```bash
# Create Oracle user
docker exec -i oracle-23ai-code-assistant sqlplus sys/CodeAssist123@FREEPDB1 as sysdba <<'EOF'
CREATE USER codeassist IDENTIFIED BY CodeAssist123;
GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE TRIGGER, CREATE SEQUENCE, UNLIMITED TABLESPACE TO codeassist;
EXIT;
EOF

# Create table
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
source venv/bin/activate
python -m src.database.seed_data
```

### 3. Start the Application

```bash
cd /Users/velloreakash/claude/oracle_task/implementation/code-assistant
source venv/bin/activate
streamlit run streamlit_app.py
```

---

## Links to Open

| Service | URL | Description |
|---------|-----|-------------|
| **Code Assistant** | http://localhost:8501 | Main Streamlit UI |
| **Jaeger UI** | http://localhost:16686 | Distributed tracing dashboard |
| **Oracle EM Express** | https://localhost:5500/em | Oracle Enterprise Manager (optional) |

---

## Demo Queries

### Category 1: Database Connections (Oracle Focus)

These queries showcase the Oracle database integration:

1. **Basic Connection**
   ```
   How do I connect to Oracle database in Python?
   ```

2. **Connection Pooling**
   ```
   Show me how to use connection pooling with Oracle database
   ```

3. **Transactions**
   ```
   How do I handle transactions in Oracle with Python?
   ```

4. **Error Handling**
   ```
   What's the best way to handle Oracle database errors in Python?
   ```

---

### Category 2: API Development (FastAPI)

These showcase web API code examples:

5. **REST API Basics**
   ```
   How do I create a REST API endpoint with FastAPI?
   ```

6. **Authentication**
   ```
   Show me JWT authentication examples for FastAPI
   ```

7. **Error Handling**
   ```
   How do I implement error handling in FastAPI?
   ```

8. **Middleware**
   ```
   How do I add middleware to a FastAPI application?
   ```

---

### Category 3: AI/LLM Integration (LangChain/LangGraph)

These highlight AI agent capabilities:

9. **LangChain Basics**
   ```
   How do I create a simple LangChain agent?
   ```

10. **LangGraph Workflows**
    ```
    Show me how to build a multi-agent workflow with LangGraph
    ```

11. **Tavily Search**
    ```
    How do I use Tavily for web search in my AI application?
    ```

12. **Tool Calling**
    ```
    How do I implement tool calling with Claude API?
    ```

---

### Category 4: Testing & Data

13. **Pytest Fixtures**
    ```
    Show me pytest fixture examples for database testing
    ```

14. **Pandas Operations**
    ```
    How do I read and process CSV files with pandas?
    ```

15. **JSON Processing**
    ```
    What's the best way to handle JSON data in Python?
    ```

---

## Demo Script (Suggested Flow)

### Part 1: Introduction (30 seconds)
- Show the Code Assistant UI
- Briefly explain the architecture diagram in README

### Part 2: Basic Query Demo (2 minutes)
1. Type: `How do I connect to Oracle database in Python?`
2. Show the response loading
3. Point out:
   - Chat message with formatted response
   - Agent Activity panel showing orchestrator progress
   - Trace Visualization with timing breakdown
   - Metrics bar at the bottom

### Part 3: Multi-Agent Explanation (1 minute)
- Explain how the query is processed:
  - Orchestrator analyzes the query
  - Doc Search Agent queries Tavily for documentation
  - Code Query Agent queries Oracle DB for code snippets
  - Results are combined into a comprehensive response

### Part 4: Jaeger Tracing (1 minute)
1. Open Jaeger UI: http://localhost:16686
2. Select service: `code-assistant`
3. Click "Find Traces"
4. Show a trace with all spans:
   - `orchestrator_analyze`
   - `doc_search_agent`
   - `code_query_agent`
   - `orchestrator_combine`

### Part 5: Additional Queries (2-3 minutes)
Try 2-3 more queries from different categories:
- `Show me how to use connection pooling with Oracle database`
- `How do I create a LangChain agent?`
- `Show me JWT authentication examples`

### Part 6: Wrap-up (30 seconds)
- Highlight key technologies used:
  - LangGraph for multi-agent orchestration
  - Oracle 23ai for code snippet storage
  - Tavily for real-time documentation search
  - OpenTelemetry + Jaeger for observability
  - Claude 3.5 Sonnet as the LLM

---

## Troubleshooting

### Docker Issues
```bash
# Check container logs
docker logs oracle-23ai-code-assistant
docker logs jaeger-code-assistant

# Restart containers
docker-compose down
docker-compose up -d
```

### Streamlit Issues
```bash
# Clear cache and restart
pkill -f streamlit
find . -name "__pycache__" -type d -exec rm -rf {} +
streamlit run streamlit_app.py
```

### Database Issues
```bash
# Check database connection
docker exec -it oracle-23ai-code-assistant sqlplus codeassist/CodeAssist123@FREEPDB1

# Check snippet count
SELECT COUNT(*) FROM code_snippets;
SELECT language, COUNT(*) FROM code_snippets GROUP BY language;
```

---

## Environment Variables

Ensure `.env` file has these values:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE=FREEPDB1
ORACLE_USER=codeassist
ORACLE_PASSWORD=CodeAssist123
```

---

## Code Snippet Categories in Database

| Category | Count | Description |
|----------|-------|-------------|
| database | 12 | Oracle connections, queries, transactions |
| api | 8 | FastAPI endpoints, middleware, error handling |
| ai | 8 | LangChain, LangGraph, Tavily examples |
| auth | 5 | JWT, OAuth, RBAC patterns |
| data | 4 | Pandas, CSV, JSON processing |
| testing | 3 | Pytest fixtures, mocking |

**Total: 40 code snippets**

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Orchestrator Agent                  │
│         (Analyzes & Routes Queries)             │
└─────────────────────────────────────────────────┘
    │                                    │
    ▼                                    ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Doc Search Agent   │    │  Code Query Agent   │
│   (Tavily API)      │    │   (Oracle 23ai)     │
└─────────────────────┘    └─────────────────────┘
    │                                    │
    └────────────────┬───────────────────┘
                     │
                     ▼
           Combined Response
                     │
                     ▼
        ┌─────────────────────┐
        │   Streamlit UI      │
        │  + Trace Viz        │
        └─────────────────────┘
```

---

## Tips for Recording

1. **Clean Browser**: Use incognito mode or clear browser cache
2. **Full Screen**: Maximize browser window for better visibility
3. **Slow Typing**: Type queries slowly for viewers to follow
4. **Pause**: After submitting a query, pause to let the response load
5. **Zoom**: Use browser zoom if text is too small
6. **Highlight**: Use mouse cursor to point at important UI elements
7. **Split Screen**: Show Streamlit and Jaeger side-by-side if possible

---

## Stop Services

```bash
# Stop Streamlit
pkill -f streamlit

# Stop Docker containers
docker-compose down

# Remove volumes (if needed for clean start)
docker-compose down -v
```

---

*Generated for Code Assistant Demo - Oracle Developer Evangelist Technical Assignment*
