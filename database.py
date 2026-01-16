import duckdb
from contextlib import contextmanager
from config import settings
import os

# Ensure database directory exists
os.makedirs(os.path.dirname(settings.DATABASE_PATH) if os.path.dirname(settings.DATABASE_PATH) else ".", exist_ok=True)

def get_connection():
    """Get a new DuckDB connection"""
    return duckdb.connect(settings.DATABASE_PATH)

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database tables"""
    with get_db() as conn:
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                email VARCHAR UNIQUE NOT NULL,
                username VARCHAR UNIQUE NOT NULL,
                hashed_password VARCHAR NOT NULL,
                full_name VARCHAR,
                plan VARCHAR DEFAULT 'starter',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Document Spaces table - Container for user's document collections
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_spaces (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                description VARCHAR,
                status VARCHAR DEFAULT 'active',
                file_count INTEGER DEFAULT 0,
                total_size_mb DOUBLE DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Datasets table - Now linked to Document Spaces
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                space_id VARCHAR,
                api_key_id VARCHAR,
                name VARCHAR NOT NULL,
                description VARCHAR,
                file_path VARCHAR NOT NULL,
                file_type VARCHAR NOT NULL,
                file_size BIGINT,
                row_count BIGINT,
                schema_json VARCHAR,
                storage_url VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # API Keys table - Now linked to Document Spaces
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                space_id VARCHAR,
                key_hash VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                scopes VARCHAR DEFAULT 'read',
                is_active BOOLEAN DEFAULT TRUE,
                last_used_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                storage_limit_mb INTEGER DEFAULT 100,
                request_limit_month INTEGER DEFAULT 10000,
                storage_used_mb DOUBLE DEFAULT 0,
                requests_this_month INTEGER DEFAULT 0
            )
        """)
        
        # API Usage tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id VARCHAR PRIMARY KEY,
                api_key_id VARCHAR NOT NULL,
                endpoint VARCHAR NOT NULL,
                method VARCHAR NOT NULL,
                status_code INTEGER,
                response_time_ms INTEGER,
                tokens_used INTEGER DEFAULT 0,
                bytes_in BIGINT DEFAULT 0,
                bytes_out BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
            )
        """)
        
        # Usage logs table (legacy)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                api_key_id VARCHAR,
                endpoint VARCHAR NOT NULL,
                method VARCHAR NOT NULL,
                status_code INTEGER,
                response_time_ms INTEGER,
                bytes_transferred BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
            )
        """)
        
        # Query history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                query_text VARCHAR NOT NULL,
                execution_time_ms INTEGER,
                row_count BIGINT,
                status VARCHAR DEFAULT 'success',
                error_message VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        print("✅ Database initialized successfully")

# Initialize on import
init_database()
