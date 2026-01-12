"""
Oracle Database Tool for LangGraph agents.
Provides database query capabilities for code snippets.

Supports two modes:
1. SQLcl MCP (if sqlcl is available)
2. Direct oracledb Python connection (fallback)
"""
import subprocess
import json
import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
import oracledb
from functools import lru_cache

# Import tracer with graceful fallback
try:
    from src.telemetry import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    # Fallback no-op tracer
    class NoOpTracer:
        def start_as_current_span(self, name, **kwargs):
            from contextlib import contextmanager
            @contextmanager
            def noop():
                class NoOpSpan:
                    def set_attribute(self, k, v): pass
                yield NoOpSpan()
            return noop()
    tracer = NoOpTracer()


def _has_sqlcl() -> bool:
    """Check if SQLcl is available on the system."""
    try:
        result = subprocess.run(
            ["sql", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_connection_params() -> dict:
    """Get Oracle connection parameters from environment."""
    return {
        "user": os.getenv("ORACLE_USER", "codeassist"),
        "password": os.getenv("ORACLE_PASSWORD", "CodeAssist123"),
        "dsn": f"{os.getenv('ORACLE_HOST', 'localhost')}:{os.getenv('ORACLE_PORT', '1521')}/{os.getenv('ORACLE_SERVICE', 'FREEPDB1')}"
    }


@lru_cache(maxsize=1)
def _get_connection_pool():
    """Get or create a connection pool (cached)."""
    params = _get_connection_params()
    try:
        return oracledb.create_pool(
            user=params["user"],
            password=params["password"],
            dsn=params["dsn"],
            min=1,
            max=5,
            increment=1
        )
    except Exception as e:
        print(f"Warning: Could not create connection pool: {e}")
        return None


class OracleDirectTool:
    """Tool for executing SQL queries via direct oracledb connection."""

    def execute_query(self, query: str, params: dict = None) -> dict:
        """
        Execute a SQL query using oracledb.

        Args:
            query: SQL query to execute
            params: Optional dict of bind parameters

        Returns:
            dict with 'success', 'data' or 'error' keys
        """
        with tracer.start_as_current_span("oracle_query") as span:
            span.set_attribute("db.system", "oracle")
            span.set_attribute("db.statement", query[:500])

            pool = _get_connection_pool()

            try:
                if pool:
                    conn = pool.acquire()
                else:
                    # Fallback to direct connection
                    conn_params = _get_connection_params()
                    conn = oracledb.connect(**conn_params)

                cursor = conn.cursor()
                cursor.execute(query, params or {})

                # Get column names
                columns = [col[0].lower() for col in cursor.description] if cursor.description else []

                # Fetch results
                rows = cursor.fetchall()

                # Convert to list of dicts
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Handle CLOB/LOB types
                        if hasattr(value, 'read'):
                            value = value.read()
                        row_dict[col] = value
                    data.append(row_dict)

                span.set_attribute("db.rows_affected", len(data))

                cursor.close()
                if pool:
                    pool.release(conn)
                else:
                    conn.close()

                return {"success": True, "data": data}

            except oracledb.Error as e:
                span.set_attribute("error", True)
                error_msg = str(e)
                return {"success": False, "error": error_msg}
            except Exception as e:
                span.set_attribute("error", True)
                return {"success": False, "error": str(e)}


class OracleSQLclTool:
    """Tool for executing SQL queries via Oracle SQLcl."""

    def __init__(self):
        self.connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """Build Oracle connection string from environment variables."""
        params = _get_connection_params()
        return f"{params['user']}/{params['password']}@{params['dsn']}"

    def execute_query(self, query: str) -> dict:
        """
        Execute a SQL query using SQLcl.

        Args:
            query: SQL query to execute

        Returns:
            dict with 'success', 'data' or 'error' keys
        """
        with tracer.start_as_current_span("oracle_sqlcl_query") as span:
            span.set_attribute("db.system", "oracle")
            span.set_attribute("db.statement", query[:500])

            try:
                # Build SQLcl command
                sqlcl_command = f"""
                SET PAGESIZE 0
                SET FEEDBACK OFF
                SET HEADING ON
                SET LINESIZE 32767
                SET LONG 1000000
                SET SQLFORMAT JSON

                {query};
                EXIT;
                """

                # Execute via subprocess
                result = subprocess.run(
                    ["sql", "-S", self.connection_string],
                    input=sqlcl_command,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    span.set_attribute("error", True)
                    return {
                        "success": False,
                        "error": result.stderr or "Query execution failed"
                    }

                # Parse JSON output
                output = result.stdout.strip()
                if output:
                    try:
                        data = json.loads(output)
                        span.set_attribute("db.rows_affected", len(data.get("items", [])))
                        return {"success": True, "data": data.get("items", [])}
                    except json.JSONDecodeError:
                        # Return raw output if not JSON
                        return {"success": True, "data": output}

                return {"success": True, "data": []}

            except subprocess.TimeoutExpired:
                span.set_attribute("error", True)
                return {"success": False, "error": "Query timeout (30s)"}
            except Exception as e:
                span.set_attribute("error", True)
                return {"success": False, "error": str(e)}


# Choose tool implementation based on availability
# Prefer direct oracledb for reliability
_oracle_tool = OracleDirectTool()


def _sanitize_input(value: str, max_length: int = 100) -> str:
    """Sanitize input to prevent SQL injection."""
    if not value:
        return ""
    # Remove dangerous characters and limit length
    sanitized = value.replace("'", "''").replace(";", "").replace("--", "")
    return sanitized[:max_length]


def _validate_limit(limit: int) -> int:
    """Validate and constrain limit parameter."""
    if not isinstance(limit, int) or limit < 1:
        return 5
    return min(limit, 20)  # Max 20 results


@tool
def search_code_snippets(
    language: Optional[str] = None,
    category: Optional[str] = None,
    framework: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 5
) -> str:
    """
    Search for code snippets in the Oracle database.

    Args:
        language: Filter by programming language (e.g., 'python', 'java')
        category: Filter by category (e.g., 'database', 'api', 'ai')
        framework: Filter by framework (e.g., 'langchain', 'fastapi')
        keyword: Search keyword in title, description, or tags
        limit: Maximum number of results (default: 5, max: 20)

    Returns:
        JSON string of matching code snippets
    """
    # Sanitize all inputs to prevent SQL injection
    safe_language = _sanitize_input(language) if language else None
    safe_category = _sanitize_input(category) if category else None
    safe_framework = _sanitize_input(framework) if framework else None
    safe_keyword = _sanitize_input(keyword) if keyword else None
    safe_limit = _validate_limit(limit)

    # Build WHERE clause with sanitized inputs
    conditions = []
    params = {}

    if safe_language:
        conditions.append("LOWER(language) = LOWER(:language)")
        params["language"] = safe_language
    if safe_category:
        conditions.append("LOWER(category) = LOWER(:category)")
        params["category"] = safe_category
    if safe_framework:
        conditions.append("LOWER(framework) LIKE LOWER(:framework)")
        params["framework"] = f"%{safe_framework}%"
    if safe_keyword:
        conditions.append("""(
            LOWER(title) LIKE LOWER(:keyword)
            OR LOWER(description) LIKE LOWER(:keyword)
            OR LOWER(tags) LIKE LOWER(:keyword)
        )""")
        params["keyword"] = f"%{safe_keyword}%"

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT id, title, description, language, framework, category, difficulty, code, tags
        FROM code_snippets
        WHERE {where_clause}
        ORDER BY created_at DESC
        FETCH FIRST {safe_limit} ROWS ONLY
    """

    result = _oracle_tool.execute_query(query, params)

    if result["success"]:
        return json.dumps(result["data"], indent=2, default=str)
    else:
        return f"Error searching snippets: {result['error']}"


@tool
def get_snippet_by_id(snippet_id: int) -> str:
    """
    Get a specific code snippet by its ID.

    Args:
        snippet_id: The unique ID of the snippet (must be positive integer)

    Returns:
        JSON string of the code snippet
    """
    # Validate snippet_id is a positive integer
    if not isinstance(snippet_id, int) or snippet_id < 1:
        return "Error: snippet_id must be a positive integer"

    query = """
        SELECT id, title, description, language, framework, category, difficulty, code, tags
        FROM code_snippets
        WHERE id = :id
    """

    result = _oracle_tool.execute_query(query, {"id": int(snippet_id)})

    if result["success"]:
        if result["data"]:
            return json.dumps(result["data"][0], indent=2, default=str)
        return "Snippet not found"
    else:
        return f"Error fetching snippet: {result['error']}"


@tool
def list_available_categories() -> str:
    """
    List all available categories and their snippet counts.

    Returns:
        JSON string of categories with counts
    """
    query = """
        SELECT category, COUNT(*) as count
        FROM code_snippets
        GROUP BY category
        ORDER BY count DESC
    """

    result = _oracle_tool.execute_query(query)

    if result["success"]:
        return json.dumps(result["data"], indent=2)
    else:
        return f"Error listing categories: {result['error']}"


@tool
def list_available_languages() -> str:
    """
    List all available programming languages and their snippet counts.

    Returns:
        JSON string of languages with counts
    """
    query = """
        SELECT language, COUNT(*) as count
        FROM code_snippets
        GROUP BY language
        ORDER BY count DESC
    """

    result = _oracle_tool.execute_query(query)

    if result["success"]:
        return json.dumps(result["data"], indent=2)
    else:
        return f"Error listing languages: {result['error']}"


# Export tools for use in agents
__all__ = [
    "search_code_snippets",
    "get_snippet_by_id",
    "list_available_categories",
    "list_available_languages"
]
