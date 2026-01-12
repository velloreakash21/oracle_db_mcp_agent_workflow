-- ================================================
-- Code Assistant Database Schema
-- Oracle 23ai
-- ================================================

-- Connect to PDB
ALTER SESSION SET CONTAINER = FREEPDB1;

-- ================================================
-- TABLE: code_snippets
-- Stores code examples and snippets
-- ================================================

CREATE TABLE code_snippets (
    -- Primary Key
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Core Fields
    title VARCHAR2(200) NOT NULL,
    description VARCHAR2(2000),

    -- Classification
    language VARCHAR2(50) NOT NULL,          -- python, java, javascript, sql
    framework VARCHAR2(100),                  -- langchain, fastapi, flask, spring
    category VARCHAR2(100),                   -- database, api, auth, ai, data
    difficulty VARCHAR2(20) DEFAULT 'intermediate'  -- beginner, intermediate, advanced
        CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),

    -- Code Content
    code CLOB NOT NULL,

    -- Metadata
    tags VARCHAR2(500),                       -- comma-separated: "oracle,connection,pooling"
    source_url VARCHAR2(500),                 -- reference URL
    author VARCHAR2(100) DEFAULT 'Code Assistant',

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================
-- INDEXES for query performance
-- ================================================

-- Language is commonly filtered
CREATE INDEX idx_snippets_language ON code_snippets(language);

-- Category for filtering by type
CREATE INDEX idx_snippets_category ON code_snippets(category);

-- Framework for specific technology queries
CREATE INDEX idx_snippets_framework ON code_snippets(framework);

-- Difficulty for skill-based filtering
CREATE INDEX idx_snippets_difficulty ON code_snippets(difficulty);

-- Full-text search on tags (basic)
CREATE INDEX idx_snippets_tags ON code_snippets(tags);

-- Composite index for common query pattern
CREATE INDEX idx_snippets_lang_cat ON code_snippets(language, category);

-- ================================================
-- TRIGGER: Auto-update timestamp
-- ================================================

CREATE OR REPLACE TRIGGER trg_snippets_updated
    BEFORE UPDATE ON code_snippets
    FOR EACH ROW
BEGIN
    :NEW.updated_at := CURRENT_TIMESTAMP;
END;
/

-- ================================================
-- VIEWS for common queries
-- ================================================

-- View: Snippet summary (without full code)
CREATE OR REPLACE VIEW v_snippet_summary AS
SELECT
    id,
    title,
    description,
    language,
    framework,
    category,
    difficulty,
    tags,
    created_at
FROM code_snippets;

-- View: Count by language
CREATE OR REPLACE VIEW v_snippets_by_language AS
SELECT
    language,
    COUNT(*) as snippet_count
FROM code_snippets
GROUP BY language
ORDER BY snippet_count DESC;

-- View: Count by category
CREATE OR REPLACE VIEW v_snippets_by_category AS
SELECT
    category,
    COUNT(*) as snippet_count
FROM code_snippets
GROUP BY category
ORDER BY snippet_count DESC;

-- ================================================
-- VERIFICATION
-- ================================================

-- Check table exists
SELECT table_name FROM user_tables WHERE table_name = 'CODE_SNIPPETS';

-- Check indexes
SELECT index_name FROM user_indexes WHERE table_name = 'CODE_SNIPPETS';

-- Describe table
DESC code_snippets;
