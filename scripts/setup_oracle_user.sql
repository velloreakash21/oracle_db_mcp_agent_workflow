-- Oracle User Setup Script
-- Run this after Oracle container is ready
-- Connect as SYSDBA: sqlplus sys/CodeAssist123@//localhost:1521/FREE as sysdba

-- Switch to the pluggable database
ALTER SESSION SET CONTAINER = FREEPDB1;

-- Create application user
CREATE USER codeassist IDENTIFIED BY CodeAssist123;

-- Grant necessary privileges
GRANT CONNECT, RESOURCE TO codeassist;
GRANT CREATE SESSION TO codeassist;
GRANT UNLIMITED TABLESPACE TO codeassist;

-- Verify user was created
SELECT username FROM dba_users WHERE username = 'CODEASSIST';

-- Show connection info
SELECT 'Connection successful. User CODEASSIST created.' AS status FROM dual;
