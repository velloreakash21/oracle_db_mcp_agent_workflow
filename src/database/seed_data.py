"""
Seed data for Code Assistant database.
Populates code_snippets table with example code.
"""
import oracledb
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CODE SNIPPETS DATA
# ============================================

SNIPPETS = [
    # ----------------------------------------
    # DATABASE CATEGORY (12 snippets)
    # ----------------------------------------
    {
        "title": "Oracle Database Connection",
        "description": "Basic connection to Oracle Database using python-oracledb",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "beginner",
        "tags": "oracle,connection,database,python",
        "code": '''import oracledb

# Basic connection
connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

print(f"Connected to Oracle {connection.version}")

# Always close connection
connection.close()'''
    },
    {
        "title": "Oracle Connection Pool",
        "description": "Create and use a connection pool for better performance in multi-threaded applications",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,connection,pool,performance,threading",
        "code": '''import oracledb

# Create connection pool
pool = oracledb.create_pool(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1",
    min=2,
    max=10,
    increment=1
)

# Acquire connection from pool
with pool.acquire() as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM employees")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

# Pool manages connections automatically
pool.close()'''
    },
    {
        "title": "Execute SQL Query with Parameters",
        "description": "Safe parameterized query execution to prevent SQL injection",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "beginner",
        "tags": "oracle,sql,query,parameters,security",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Use bind variables (NEVER string concatenation!)
employee_id = 100
cursor.execute(
    "SELECT first_name, last_name FROM employees WHERE employee_id = :id",
    {"id": employee_id}
)

row = cursor.fetchone()
if row:
    print(f"Employee: {row[0]} {row[1]}")

cursor.close()
connection.close()'''
    },
    {
        "title": "Insert Data with RETURNING Clause",
        "description": "Insert data and return generated values like auto-increment IDs",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,insert,returning,identity",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Variable to hold returned ID
new_id = cursor.var(oracledb.NUMBER)

cursor.execute("""
    INSERT INTO employees (first_name, last_name, email)
    VALUES (:fname, :lname, :email)
    RETURNING employee_id INTO :new_id
""", {
    "fname": "John",
    "lname": "Doe",
    "email": "john.doe@example.com",
    "new_id": new_id
})

connection.commit()
print(f"Inserted employee with ID: {new_id.getvalue()[0]}")

cursor.close()
connection.close()'''
    },
    {
        "title": "Batch Insert with executemany",
        "description": "Efficiently insert multiple rows in a single operation",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,insert,batch,bulk,performance",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Data to insert
data = [
    ("Alice", "Smith", "alice@example.com"),
    ("Bob", "Johnson", "bob@example.com"),
    ("Carol", "Williams", "carol@example.com"),
]

# Batch insert - much faster than individual inserts
cursor.executemany(
    """INSERT INTO employees (first_name, last_name, email)
       VALUES (:1, :2, :3)""",
    data
)

connection.commit()
print(f"Inserted {cursor.rowcount} rows")

cursor.close()
connection.close()'''
    },
    {
        "title": "Transaction Management",
        "description": "Handle database transactions with commit and rollback",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,transaction,commit,rollback,acid",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

try:
    cursor = connection.cursor()

    # First operation
    cursor.execute(
        "UPDATE accounts SET balance = balance - :amount WHERE id = :id",
        {"amount": 100, "id": 1}
    )

    # Second operation
    cursor.execute(
        "UPDATE accounts SET balance = balance + :amount WHERE id = :id",
        {"amount": 100, "id": 2}
    )

    # Both succeed - commit transaction
    connection.commit()
    print("Transfer successful!")

except oracledb.Error as e:
    # Something failed - rollback all changes
    connection.rollback()
    print(f"Transfer failed, rolled back: {e}")

finally:
    cursor.close()
    connection.close()'''
    },
    {
        "title": "Call Stored Procedure",
        "description": "Execute an Oracle stored procedure with input/output parameters",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "advanced",
        "tags": "oracle,procedure,plsql,stored-procedure",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Output variable for result
result = cursor.var(oracledb.STRING, 100)

# Call procedure with IN and OUT parameters
cursor.callproc("get_employee_name", [100, result])

print(f"Employee name: {result.getvalue()}")

# Alternative: Call function
name = cursor.callfunc("get_employee_name_fn", oracledb.STRING, [100])
print(f"Employee name: {name}")

cursor.close()
connection.close()'''
    },
    {
        "title": "Fetch Large Result Sets with Batching",
        "description": "Efficiently fetch large datasets using fetchmany",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,fetch,performance,pagination,batch",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Set array size for optimal fetching
cursor.arraysize = 1000

cursor.execute("SELECT * FROM large_table")

# Fetch in batches to manage memory
batch_size = 1000
total_rows = 0

while True:
    rows = cursor.fetchmany(batch_size)
    if not rows:
        break

    total_rows += len(rows)
    # Process batch
    for row in rows:
        pass  # Your processing here

    print(f"Processed {total_rows} rows...")

print(f"Total rows processed: {total_rows}")

cursor.close()
connection.close()'''
    },
    {
        "title": "Working with CLOB Data",
        "description": "Read and write large text data using CLOB columns",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,clob,lob,text,large-object",
        "code": '''import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Insert CLOB data
large_text = "A" * 100000  # 100KB of text

cursor.execute(
    "INSERT INTO documents (id, content) VALUES (:id, :content)",
    {"id": 1, "content": large_text}
)
connection.commit()

# Read CLOB data
cursor.execute("SELECT content FROM documents WHERE id = :id", {"id": 1})
row = cursor.fetchone()

if row:
    clob_data = row[0].read()  # Read full CLOB
    print(f"Retrieved {len(clob_data)} characters")

cursor.close()
connection.close()'''
    },
    {
        "title": "Context Manager Pattern",
        "description": "Use context managers for automatic resource cleanup",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "beginner",
        "tags": "oracle,context-manager,cleanup,best-practice",
        "code": '''import oracledb

# Connection context manager - auto-closes
with oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
) as connection:

    # Cursor context manager - auto-closes
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM employees")
        count = cursor.fetchone()[0]
        print(f"Total employees: {count}")

    # Cursor closed here

# Connection closed here
# No need for explicit close() calls!'''
    },
    {
        "title": "Error Handling for Oracle",
        "description": "Properly catch and handle Oracle database errors",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "intermediate",
        "tags": "oracle,error,exception,handling",
        "code": '''import oracledb

try:
    connection = oracledb.connect(
        user="your_user",
        password="your_password",
        dsn="localhost:1521/FREEPDB1"
    )

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM nonexistent_table")

except oracledb.DatabaseError as e:
    error, = e.args
    print(f"Oracle Error Code: {error.code}")
    print(f"Error Message: {error.message}")

    # Handle specific errors
    if error.code == 942:  # ORA-00942: table or view does not exist
        print("Table not found - check table name")
    elif error.code == 1017:  # ORA-01017: invalid username/password
        print("Authentication failed - check credentials")

except oracledb.InterfaceError as e:
    print(f"Interface Error: {e}")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals():
        connection.close()'''
    },
    {
        "title": "Oracle JSON Query",
        "description": "Query JSON data stored in Oracle Database",
        "language": "python",
        "framework": "oracledb",
        "category": "database",
        "difficulty": "advanced",
        "tags": "oracle,json,nosql,document",
        "code": '''import oracledb
import json

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Insert JSON data
user_data = {"name": "John", "age": 30, "skills": ["Python", "SQL"]}
cursor.execute(
    "INSERT INTO users_json (id, data) VALUES (:id, :data)",
    {"id": 1, "data": json.dumps(user_data)}
)

# Query JSON using Oracle JSON functions
cursor.execute("""
    SELECT u.data.name, u.data.age
    FROM users_json u
    WHERE JSON_EXISTS(u.data, '$.skills[*]?(@ == "Python")')
""")

for row in cursor:
    print(f"Name: {row[0]}, Age: {row[1]}")

cursor.close()
connection.close()'''
    },

    # ----------------------------------------
    # API CATEGORY (8 snippets)
    # ----------------------------------------
    {
        "title": "FastAPI Basic Endpoint",
        "description": "Create a simple REST API endpoint with FastAPI",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "beginner",
        "tags": "fastapi,rest,api,endpoint",
        "code": '''from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: Item):
    return item

# Run with: uvicorn main:app --reload'''
    },
    {
        "title": "FastAPI with Database Dependency",
        "description": "Inject database connection using FastAPI dependencies",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "intermediate",
        "tags": "fastapi,dependency,database,injection",
        "code": '''from fastapi import FastAPI, Depends
from contextlib import contextmanager
import oracledb

app = FastAPI()

# Database connection pool
pool = oracledb.create_pool(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1",
    min=2, max=10
)

# Dependency to get database connection
def get_db():
    connection = pool.acquire()
    try:
        yield connection
    finally:
        pool.release(connection)

@app.get("/users/{user_id}")
async def get_user(user_id: int, db = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, name, email FROM users WHERE id = :id",
        {"id": user_id}
    )
    row = cursor.fetchone()
    cursor.close()

    if row:
        return {"id": row[0], "name": row[1], "email": row[2]}
    return {"error": "User not found"}'''
    },
    {
        "title": "FastAPI Error Handling",
        "description": "Custom exception handlers for API errors",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "intermediate",
        "tags": "fastapi,error,exception,handling",
        "code": '''from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Item not found",
            "item_id": exc.item_id,
            "detail": f"Item with ID {exc.item_id} does not exist"
        }
    )

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    items = {1: "Apple", 2: "Banana"}

    if item_id not in items:
        raise ItemNotFoundError(item_id)

    return {"item_id": item_id, "name": items[item_id]}'''
    },
    {
        "title": "FastAPI Request Validation",
        "description": "Validate request data with Pydantic models",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "intermediate",
        "tags": "fastapi,validation,pydantic,request",
        "code": '''from fastapi import FastAPI
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional
from datetime import date

app = FastAPI()

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=18, le=120)
    birth_date: Optional[date] = None

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 25,
                "birth_date": "1999-01-15"
            }
        }

@app.post("/users/")
async def create_user(user: UserCreate):
    return {"message": "User created", "user": user}'''
    },
    {
        "title": "FastAPI Background Tasks",
        "description": "Run tasks in background after returning response",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "intermediate",
        "tags": "fastapi,background,async,task",
        "code": '''from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def send_email(email: str, message: str):
    """Simulate sending email (slow operation)"""
    time.sleep(5)  # Simulate delay
    print(f"Email sent to {email}: {message}")

def write_log(message: str):
    """Write to log file"""
    with open("app.log", "a") as f:
        f.write(f"{message}\\n")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    # Add tasks to run after response
    background_tasks.add_task(send_email, email, "Welcome!")
    background_tasks.add_task(write_log, f"Notification sent to {email}")

    # Return immediately - tasks run in background
    return {"message": "Notification scheduled"}'''
    },
    {
        "title": "FastAPI CORS Configuration",
        "description": "Enable Cross-Origin Resource Sharing for frontend access",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "beginner",
        "tags": "fastapi,cors,security,frontend",
        "code": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",      # React dev server
    "http://localhost:8080",      # Vue dev server
    "https://myapp.example.com",  # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/api/data")
async def get_data():
    return {"data": "This is accessible from allowed origins"}'''
    },
    {
        "title": "FastAPI Rate Limiting",
        "description": "Implement rate limiting to prevent API abuse",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "advanced",
        "tags": "fastapi,rate-limit,security,throttle",
        "code": '''from fastapi import FastAPI, Request, HTTPException
from collections import defaultdict
import time

app = FastAPI()

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]

        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False

        self.requests[client_ip].append(now)
        return True

limiter = RateLimiter(requests_per_minute=10)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host

    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    return await call_next(request)

@app.get("/api/data")
async def get_data():
    return {"data": "Rate limited endpoint"}'''
    },
    {
        "title": "FastAPI File Upload",
        "description": "Handle file uploads with validation",
        "language": "python",
        "framework": "fastapi",
        "category": "api",
        "difficulty": "intermediate",
        "tags": "fastapi,file,upload,multipart",
        "code": '''from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import shutil
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = ["image/jpeg", "image/png", "application/pdf"]
MAX_SIZE = 5 * 1024 * 1024  # 5MB

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "File type not allowed")

    # Read and check size
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, "File too large (max 5MB)")

    # Save file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "filename": file.filename,
        "size": len(contents),
        "type": file.content_type
    }

@app.post("/upload-multiple/")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        results.append({"filename": file.filename, "size": len(contents)})
    return results'''
    },

    # ----------------------------------------
    # AI CATEGORY (8 snippets)
    # ----------------------------------------
    {
        "title": "LangChain Basic Chat",
        "description": "Simple chat completion with LangChain and Claude",
        "language": "python",
        "framework": "langchain",
        "category": "ai",
        "difficulty": "beginner",
        "tags": "langchain,claude,chat,llm",
        "code": '''from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key="your-api-key"
)

# Simple chat
messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="How do I connect to Oracle database in Python?")
]

response = llm.invoke(messages)
print(response.content)'''
    },
    {
        "title": "LangChain Tool Calling",
        "description": "Create and use tools with LangChain agents",
        "language": "python",
        "framework": "langchain",
        "category": "ai",
        "difficulty": "intermediate",
        "tags": "langchain,tools,function-calling,agent",
        "code": '''from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def search_database(query: str) -> str:
    """Search the code snippets database for relevant examples."""
    # In real implementation, query your database
    return f"Found 3 results for: {query}"

@tool
def get_documentation(topic: str) -> str:
    """Get documentation for a programming topic."""
    return f"Documentation for {topic}: ..."

# Create LLM with tools
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools([search_database, get_documentation])

# Use tools
messages = [HumanMessage(content="Find Python examples for Oracle connection")]
response = llm_with_tools.invoke(messages)

# Check if tool was called
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")'''
    },
    {
        "title": "LangGraph Simple Agent",
        "description": "Build a basic agent with LangGraph state management",
        "language": "python",
        "framework": "langgraph",
        "category": "ai",
        "difficulty": "intermediate",
        "tags": "langgraph,agent,state,graph",
        "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

# Define nodes
def analyze_query(state: AgentState) -> AgentState:
    """Analyze the user query"""
    return {
        "messages": ["Analyzed query"],
        "next_step": "search"
    }

def search_info(state: AgentState) -> AgentState:
    """Search for information"""
    return {
        "messages": ["Found relevant info"],
        "next_step": "respond"
    }

def generate_response(state: AgentState) -> AgentState:
    """Generate final response"""
    return {
        "messages": ["Generated response"],
        "next_step": "end"
    }

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_query)
workflow.add_node("search", search_info)
workflow.add_node("respond", generate_response)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "search")
workflow.add_edge("search", "respond")
workflow.add_edge("respond", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [], "next_step": ""})
print(result)'''
    },
    {
        "title": "LangGraph Multi-Agent System",
        "description": "Create a multi-agent system with orchestrator pattern",
        "language": "python",
        "framework": "langgraph",
        "category": "ai",
        "difficulty": "advanced",
        "tags": "langgraph,multi-agent,orchestrator,routing",
        "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    query: str
    search_results: str
    db_results: str
    final_response: str

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def orchestrator(state: State) -> Literal["search_agent", "db_agent", "end"]:
    """Route to appropriate agent based on query"""
    query = state["query"].lower()
    if "search" in query or "find" in query:
        return "search_agent"
    elif "database" in query or "sql" in query:
        return "db_agent"
    return "end"

def search_agent(state: State) -> State:
    """Search documentation"""
    return {"search_results": f"Search results for: {state['query']}"}

def db_agent(state: State) -> State:
    """Query database"""
    return {"db_results": f"DB results for: {state['query']}"}

def synthesizer(state: State) -> State:
    """Combine results into final response"""
    combined = f"Search: {state.get('search_results', 'N/A')}, DB: {state.get('db_results', 'N/A')}"
    return {"final_response": combined}

# Build multi-agent graph
workflow = StateGraph(State)

workflow.add_node("search_agent", search_agent)
workflow.add_node("db_agent", db_agent)
workflow.add_node("synthesizer", synthesizer)

workflow.set_conditional_entry_point(orchestrator)
workflow.add_edge("search_agent", "synthesizer")
workflow.add_edge("db_agent", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()'''
    },
    {
        "title": "Tavily Web Search",
        "description": "Search the web using Tavily API",
        "language": "python",
        "framework": "tavily",
        "category": "ai",
        "difficulty": "beginner",
        "tags": "tavily,search,web,api",
        "code": '''from tavily import TavilyClient

client = TavilyClient(api_key="your-tavily-api-key")

# Basic search
response = client.search(
    query="Python Oracle database connection best practices",
    search_depth="advanced",
    max_results=5
)

for result in response["results"]:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")

# Search with context for AI
context = client.get_search_context(
    query="How to use connection pooling in Oracle",
    max_tokens=4000
)
print(context)  # Ready to use in LLM prompt'''
    },
    {
        "title": "OpenTelemetry LLM Tracing",
        "description": "Trace LLM calls with OpenTelemetry",
        "language": "python",
        "framework": "opentelemetry",
        "category": "ai",
        "difficulty": "intermediate",
        "tags": "opentelemetry,tracing,observability,llm",
        "code": '''from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from langchain_anthropic import ChatAnthropic

# Setup tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Traced LLM call
def traced_llm_call(query: str) -> str:
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("llm.model", "claude-sonnet-4-20250514")
        span.set_attribute("llm.query", query)

        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        response = llm.invoke(query)

        span.set_attribute("llm.response_length", len(response.content))
        return response.content

result = traced_llm_call("Explain Python decorators")'''
    },
    {
        "title": "Prompt Template with Variables",
        "description": "Create reusable prompt templates",
        "language": "python",
        "framework": "langchain",
        "category": "ai",
        "difficulty": "beginner",
        "tags": "langchain,prompt,template,variables",
        "code": '''from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic

# Create template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant. Be {tone} in your responses."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

# Create chain
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
chain = template | llm

# Use template
response = chain.invoke({
    "role": "coding",
    "tone": "concise and technical",
    "history": [],
    "query": "How do I handle errors in Python?"
})

print(response.content)'''
    },
    {
        "title": "Streaming LLM Response",
        "description": "Stream LLM responses for better UX",
        "language": "python",
        "framework": "langchain",
        "category": "ai",
        "difficulty": "intermediate",
        "tags": "langchain,streaming,async,llm",
        "code": '''from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import asyncio

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Synchronous streaming
def stream_response(query: str):
    for chunk in llm.stream([HumanMessage(content=query)]):
        print(chunk.content, end="", flush=True)
    print()  # Newline at end

# Async streaming
async def astream_response(query: str):
    async for chunk in llm.astream([HumanMessage(content=query)]):
        print(chunk.content, end="", flush=True)
    print()

# Usage
stream_response("Write a haiku about coding")

# Async usage
asyncio.run(astream_response("Write a haiku about debugging"))'''
    },

    # ----------------------------------------
    # AUTH CATEGORY (5 snippets)
    # ----------------------------------------
    {
        "title": "JWT Token Generation",
        "description": "Generate and validate JWT tokens for API authentication",
        "language": "python",
        "framework": "pyjwt",
        "category": "auth",
        "difficulty": "intermediate",
        "tags": "jwt,token,authentication,security",
        "code": '''import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key-keep-it-safe"
ALGORITHM = "HS256"

def create_access_token(user_id: int, email: str) -> str:
    """Create a JWT access token"""
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict | None:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except jwt.InvalidTokenError:
        print("Invalid token")
        return None

# Usage
token = create_access_token(user_id=123, email="user@example.com")
print(f"Token: {token}")

decoded = verify_token(token)
print(f"Decoded: {decoded}")'''
    },
    {
        "title": "Password Hashing with bcrypt",
        "description": "Securely hash and verify passwords",
        "language": "python",
        "framework": "bcrypt",
        "category": "auth",
        "difficulty": "beginner",
        "tags": "password,hash,bcrypt,security",
        "code": '''import bcrypt

def hash_password(password: str) -> str:
    """Hash a password for storage"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed.encode('utf-8')
    )

# Usage
password = "my_secure_password_123"
hashed = hash_password(password)
print(f"Hashed: {hashed}")

# Verify correct password
is_valid = verify_password("my_secure_password_123", hashed)
print(f"Valid password: {is_valid}")  # True

# Verify wrong password
is_valid = verify_password("wrong_password", hashed)
print(f"Wrong password: {is_valid}")  # False'''
    },
    {
        "title": "FastAPI OAuth2 Password Flow",
        "description": "Implement OAuth2 password authentication in FastAPI",
        "language": "python",
        "framework": "fastapi",
        "category": "auth",
        "difficulty": "advanced",
        "tags": "fastapi,oauth2,authentication,api",
        "code": '''from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"

# Fake user database
fake_users = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": "fakehashedsecret",
        "email": "john@example.com"
    }
}

class User(BaseModel):
    username: str
    email: str

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        user = fake_users.get(username)
        if user:
            return User(username=user["username"], email=user["email"])
    except jwt.InvalidTokenError:
        pass
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users.get(form_data.username)
    if not user or form_data.password != "secret":
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = jwt.encode({"sub": user["username"]}, SECRET_KEY)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user'''
    },
    {
        "title": "API Key Authentication",
        "description": "Simple API key authentication for FastAPI",
        "language": "python",
        "framework": "fastapi",
        "category": "auth",
        "difficulty": "beginner",
        "tags": "fastapi,api-key,authentication,header",
        "code": '''from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI()

API_KEY_NAME = "X-API-Key"
API_KEYS = {"secret-api-key-1", "secret-api-key-2"}

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key header missing"
        )
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

@app.get("/protected")
async def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "Access granted", "key_used": api_key[:8] + "..."}

@app.get("/public")
async def public_route():
    return {"message": "This is public"}'''
    },
    {
        "title": "Role-Based Access Control",
        "description": "Implement RBAC for API endpoints",
        "language": "python",
        "framework": "fastapi",
        "category": "auth",
        "difficulty": "advanced",
        "tags": "fastapi,rbac,authorization,roles",
        "code": '''from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from enum import Enum
from typing import List

app = FastAPI()

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

# Simulated user with roles
class User:
    def __init__(self, username: str, roles: List[Role]):
        self.username = username
        self.roles = roles

def get_current_user() -> User:
    # In real app, decode from JWT token
    return User("john", [Role.USER])

def require_roles(allowed_roles: List[Role]):
    """Dependency to check user roles"""
    def role_checker(user: User = Depends(get_current_user)):
        for role in user.roles:
            if role in allowed_roles:
                return user
        raise HTTPException(
            status_code=403,
            detail=f"Requires one of roles: {[r.value for r in allowed_roles]}"
        )
    return role_checker

@app.get("/admin-only")
async def admin_only(user: User = Depends(require_roles([Role.ADMIN]))):
    return {"message": "Welcome admin!", "user": user.username}

@app.get("/user-area")
async def user_area(user: User = Depends(require_roles([Role.USER, Role.ADMIN]))):
    return {"message": "Welcome user!", "user": user.username}'''
    },

    # ----------------------------------------
    # DATA CATEGORY (4 snippets)
    # ----------------------------------------
    {
        "title": "Pandas DataFrame from Oracle",
        "description": "Load Oracle query results into Pandas DataFrame",
        "language": "python",
        "framework": "pandas",
        "category": "data",
        "difficulty": "beginner",
        "tags": "pandas,oracle,dataframe,sql",
        "code": '''import pandas as pd
import oracledb

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

# Method 1: Using read_sql_query
df = pd.read_sql_query(
    "SELECT * FROM employees WHERE department_id = :dept",
    connection,
    params={"dept": 10}
)

print(df.head())
print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")

# Method 2: From cursor
cursor = connection.cursor()
cursor.execute("SELECT employee_id, first_name, salary FROM employees")
columns = [col[0] for col in cursor.description]
data = cursor.fetchall()
df2 = pd.DataFrame(data, columns=columns)

cursor.close()
connection.close()'''
    },
    {
        "title": "DataFrame to Oracle Table",
        "description": "Write Pandas DataFrame to Oracle database table",
        "language": "python",
        "framework": "pandas",
        "category": "data",
        "difficulty": "intermediate",
        "tags": "pandas,oracle,insert,bulk",
        "code": '''import pandas as pd
import oracledb

# Sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

connection = oracledb.connect(
    user="your_user",
    password="your_password",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS people (
        name VARCHAR2(100),
        age NUMBER,
        city VARCHAR2(100)
    )
""")

# Convert DataFrame to list of tuples
data = [tuple(row) for row in df.values]

# Bulk insert
cursor.executemany(
    "INSERT INTO people (name, age, city) VALUES (:1, :2, :3)",
    data
)

connection.commit()
print(f"Inserted {cursor.rowcount} rows")

cursor.close()
connection.close()'''
    },
    {
        "title": "JSON Processing with Pandas",
        "description": "Parse and process JSON data with Pandas",
        "language": "python",
        "framework": "pandas",
        "category": "data",
        "difficulty": "intermediate",
        "tags": "pandas,json,parsing,transform",
        "code": '''import pandas as pd
import json

# From JSON string
json_string = \'\'\'[
    {"name": "Alice", "scores": [85, 90, 88]},
    {"name": "Bob", "scores": [78, 85, 92]}
]\'\'\'

df = pd.read_json(json_string)
print(df)

# Normalize nested JSON
data = [
    {"name": "Alice", "info": {"age": 25, "city": "NYC"}},
    {"name": "Bob", "info": {"age": 30, "city": "LA"}}
]

df_flat = pd.json_normalize(data)
print(df_flat)

# DataFrame to JSON
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

# Different orientations
print(df.to_json(orient='records'))  # [{"name":"Alice","age":25},...]
print(df.to_json(orient='index'))    # {"0":{"name":"Alice",...},...}'''
    },
    {
        "title": "CSV File Operations",
        "description": "Read and write CSV files with Pandas",
        "language": "python",
        "framework": "pandas",
        "category": "data",
        "difficulty": "beginner",
        "tags": "pandas,csv,file,io",
        "code": '''import pandas as pd

# Read CSV with options
df = pd.read_csv(
    'data.csv',
    encoding='utf-8',
    sep=',',
    header=0,
    dtype={'id': int, 'name': str},
    parse_dates=['created_at'],
    na_values=['', 'NULL', 'N/A']
)

print(f"Loaded {len(df)} rows")
print(df.dtypes)

# Filter and transform
df_filtered = df[df['age'] > 25]
df_filtered['age_group'] = df_filtered['age'].apply(
    lambda x: 'young' if x < 30 else 'senior'
)

# Write to CSV
df_filtered.to_csv(
    'output.csv',
    index=False,
    encoding='utf-8',
    date_format='%Y-%m-%d'
)

# Read in chunks for large files
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed = chunk[chunk['status'] == 'active']
    # Append to output
    processed.to_csv('output.csv', mode='a', header=False, index=False)'''
    },

    # ----------------------------------------
    # TESTING CATEGORY (3 snippets)
    # ----------------------------------------
    {
        "title": "Pytest Basic Test",
        "description": "Write basic unit tests with pytest",
        "language": "python",
        "framework": "pytest",
        "category": "testing",
        "difficulty": "beginner",
        "tags": "pytest,unit-test,testing",
        "code": '''import pytest

# Function to test
def add(a: int, b: int) -> int:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Basic test
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

# Test with expected exception
def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

# Parametrized test
@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5),
    (9, 3, 3),
    (7, 2, 3.5),
])
def test_divide(a, b, expected):
    assert divide(a, b) == expected

# Run with: pytest test_file.py -v'''
    },
    {
        "title": "Pytest Fixtures",
        "description": "Use fixtures for test setup and teardown",
        "language": "python",
        "framework": "pytest",
        "category": "testing",
        "difficulty": "intermediate",
        "tags": "pytest,fixtures,setup,teardown",
        "code": '''import pytest

# Simple fixture
@pytest.fixture
def sample_data():
    return {"name": "Test", "value": 42}

# Fixture with setup and teardown
@pytest.fixture
def database_connection():
    # Setup
    conn = {"connected": True, "host": "localhost"}
    print("\\nConnecting to database...")

    yield conn  # This is what the test receives

    # Teardown
    print("\\nClosing database connection...")
    conn["connected"] = False

# Fixture with scope
@pytest.fixture(scope="module")
def expensive_resource():
    """Created once per module, not per test"""
    print("\\nCreating expensive resource...")
    return {"resource": "expensive"}

# Using fixtures in tests
def test_with_sample_data(sample_data):
    assert sample_data["name"] == "Test"
    assert sample_data["value"] == 42

def test_with_database(database_connection):
    assert database_connection["connected"] is True

# Fixture that uses another fixture
@pytest.fixture
def user_with_data(sample_data):
    return {"user_id": 1, **sample_data}

def test_user(user_with_data):
    assert user_with_data["user_id"] == 1
    assert user_with_data["name"] == "Test"'''
    },
    {
        "title": "Mocking with pytest",
        "description": "Mock external dependencies in tests",
        "language": "python",
        "framework": "pytest",
        "category": "testing",
        "difficulty": "intermediate",
        "tags": "pytest,mock,unittest,testing",
        "code": '''import pytest
from unittest.mock import Mock, patch, MagicMock

# Function that calls external API
def fetch_user(user_id: int) -> dict:
    import requests
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

# Mock the requests.get
@patch('requests.get')
def test_fetch_user(mock_get):
    # Configure mock response
    mock_response = Mock()
    mock_response.json.return_value = {"id": 1, "name": "John"}
    mock_get.return_value = mock_response

    # Call function
    result = fetch_user(1)

    # Assertions
    assert result["name"] == "John"
    mock_get.assert_called_once_with("https://api.example.com/users/1")

# Using MagicMock for complex objects
def test_with_magic_mock():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [
        {"id": 1}, {"id": 2}
    ]

    result = mock_db.query("users").filter(active=True).all()
    assert len(result) == 2

# Fixture with mock
@pytest.fixture
def mock_api():
    with patch('mymodule.api_client') as mock:
        mock.get_data.return_value = {"status": "ok"}
        yield mock

def test_with_mock_fixture(mock_api):
    # mock_api is already configured
    pass'''
    },
]


def get_connection():
    """Get Oracle database connection."""
    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "codeassist"),
        password=os.getenv("ORACLE_PASSWORD", "CodeAssist123"),
        dsn=f"{os.getenv('ORACLE_HOST', 'localhost')}:{os.getenv('ORACLE_PORT', '1521')}/{os.getenv('ORACLE_SERVICE', 'FREEPDB1')}"
    )


def clear_existing_data(cursor):
    """Clear existing snippets (for idempotent runs)."""
    cursor.execute("DELETE FROM code_snippets")
    print("Cleared existing data")


def insert_snippets(cursor, snippets):
    """Insert all snippets into database."""
    insert_sql = """
        INSERT INTO code_snippets
        (title, description, language, framework, category, difficulty, tags, code)
        VALUES (:title, :description, :language, :framework, :category, :difficulty, :tags, :code)
    """

    for snippet in snippets:
        cursor.execute(insert_sql, snippet)

    print(f"Inserted {len(snippets)} snippets")


def verify_data(cursor):
    """Print summary of inserted data."""
    cursor.execute("""
        SELECT category, COUNT(*) as cnt
        FROM code_snippets
        GROUP BY category
        ORDER BY cnt DESC
    """)

    print("\nSnippets by category:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]}")

    cursor.execute("SELECT COUNT(*) FROM code_snippets")
    total = cursor.fetchone()[0]
    print(f"\nTotal snippets: {total}")


def main():
    """Main function to seed database."""
    print("Starting database seed...")

    conn = get_connection()
    cursor = conn.cursor()

    try:
        clear_existing_data(cursor)
        insert_snippets(cursor, SNIPPETS)
        conn.commit()
        verify_data(cursor)
        print("\nSeed completed successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
