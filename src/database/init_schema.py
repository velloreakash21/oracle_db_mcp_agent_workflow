"""
Initialize database schema for Code Assistant.
Run this script to create tables in Oracle DB.
"""
import oracledb
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Get Oracle database connection."""
    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "codeassist"),
        password=os.getenv("ORACLE_PASSWORD", "CodeAssist123"),
        dsn=f"{os.getenv('ORACLE_HOST', 'localhost')}:{os.getenv('ORACLE_PORT', '1521')}/{os.getenv('ORACLE_SERVICE', 'FREEPDB1')}"
    )


def execute_schema():
    """Execute schema SQL file."""
    schema_path = Path(__file__).parent / "schema.sql"

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    conn = get_connection()
    cursor = conn.cursor()

    # Split by semicolons, but handle PL/SQL blocks (triggers) separately
    # PL/SQL blocks end with '/' on a separate line
    plsql_pattern = r'CREATE\s+OR\s+REPLACE\s+TRIGGER.*?END;\s*/\s*'
    plsql_blocks = re.findall(plsql_pattern, schema_sql, re.DOTALL | re.IGNORECASE)

    # Remove PL/SQL blocks from main SQL
    remaining_sql = re.sub(plsql_pattern, '', schema_sql, flags=re.DOTALL | re.IGNORECASE)

    # Split remaining SQL by semicolons
    statements = [s.strip() for s in remaining_sql.split(';') if s.strip()]

    # Execute regular statements
    for stmt in statements:
        # Skip comments-only statements and ALTER SESSION (not needed from app user)
        if stmt.startswith('--') or not stmt or 'ALTER SESSION' in stmt.upper():
            continue
        # Skip DESC commands (Oracle SQL*Plus specific)
        if stmt.strip().upper().startswith('DESC '):
            continue
        try:
            cursor.execute(stmt)
            print(f"✓ Executed: {stmt[:60]}...")
        except oracledb.Error as e:
            error_msg = str(e)
            if 'ORA-00955' in error_msg:  # Object already exists
                print(f"⚠ Already exists: {stmt[:60]}...")
            else:
                print(f"✗ Error: {error_msg}")

    # Execute PL/SQL blocks (triggers)
    for block in plsql_blocks:
        try:
            # Remove the trailing '/' for oracledb
            clean_block = block.rstrip().rstrip('/')
            cursor.execute(clean_block)
            print(f"✓ Created trigger successfully")
        except oracledb.Error as e:
            error_msg = str(e)
            if 'ORA-04081' in error_msg:  # Trigger already exists
                print(f"⚠ Trigger already exists")
            else:
                print(f"✗ Error creating trigger: {error_msg}")

    conn.commit()
    cursor.close()
    conn.close()
    print("\n✓ Schema initialization complete!")


def verify_schema():
    """Verify schema was created correctly."""
    conn = get_connection()
    cursor = conn.cursor()

    print("\n--- Schema Verification ---")

    # Check table exists
    cursor.execute("SELECT table_name FROM user_tables WHERE table_name = 'CODE_SNIPPETS'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")

    # Check indexes
    cursor.execute("SELECT index_name FROM user_indexes WHERE table_name = 'CODE_SNIPPETS'")
    indexes = cursor.fetchall()
    print(f"Indexes: {[i[0] for i in indexes]}")

    # Count records
    cursor.execute("SELECT COUNT(*) FROM code_snippets")
    count = cursor.fetchone()[0]
    print(f"Records: {count}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    print("Initializing Code Assistant database schema...")
    execute_schema()
    verify_schema()
