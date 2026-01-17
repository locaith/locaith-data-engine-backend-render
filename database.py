import duckdb
from contextlib import contextmanager
from config import settings
import os

# Ensure database directory exists
os.makedirs(os.path.dirname(settings.DATABASE_PATH) if os.path.dirname(settings.DATABASE_PATH) else ".", exist_ok=True)

def get_connection():
    """Get a new DuckDB connection"""
    return duckdb.connect(settings.DATABASE_PATH)

def ensure_column_exists(conn, table_name, column_name, column_type):
    """Ensure a column exists in a table, add it if not"""
    try:
        # Check if column exists
        columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        column_names = [col[1] for col in columns]
        
        if column_name not in column_names:
            print(f"[Database] Adding column {column_name} to table {table_name}...")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    except Exception as e:
        print(f"[Database] Error ensuring column {column_name} in {table_name}: {e}")

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
                status VARCHAR DEFAULT 'success',
                error_message VARCHAR,
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
        
        # ============ GOLD LAYER TABLES ============
        # Gold tables - Structured data extracted from documents
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_tables (
                id VARCHAR PRIMARY KEY,
                dataset_id VARCHAR NOT NULL,
                space_id VARCHAR,
                table_name VARCHAR NOT NULL,
                source_file VARCHAR,
                source_type VARCHAR,
                schema_json VARCHAR,
                row_count INTEGER DEFAULT 0,
                column_count INTEGER DEFAULT 0,
                quality_score DOUBLE DEFAULT 0,
                quality_level VARCHAR DEFAULT 'bronze',
                parquet_path VARCHAR,
                is_queryable BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Gold columns - Column metadata for each gold table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_columns (
                id VARCHAR PRIMARY KEY,
                gold_table_id VARCHAR NOT NULL,
                column_name VARCHAR NOT NULL,
                column_type VARCHAR NOT NULL,
                sql_type VARCHAR NOT NULL,
                is_nullable BOOLEAN DEFAULT TRUE,
                is_primary_key BOOLEAN DEFAULT FALSE,
                sample_values VARCHAR,
                valid_percentage DOUBLE DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gold_table_id) REFERENCES gold_tables(id)
            )
        """)
        
        # Query cache - For faster repeated queries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id VARCHAR PRIMARY KEY,
                space_id VARCHAR NOT NULL,
                query_hash VARCHAR NOT NULL,
                query_text VARCHAR NOT NULL,
                result_json VARCHAR,
                hit_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_hit_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ============ MIGRATIONS ============
        # Ensure status columns exist (for background processing)
        ensure_column_exists(conn, "datasets", "status", "VARCHAR DEFAULT 'success'")
        ensure_column_exists(conn, "datasets", "error_message", "VARCHAR")
        
        # Ensure document_spaces has all columns
        ensure_column_exists(conn, "document_spaces", "status", "VARCHAR DEFAULT 'active'")
        ensure_column_exists(conn, "document_spaces", "file_count", "INTEGER DEFAULT 0")
        ensure_column_exists(conn, "document_spaces", "total_size_mb", "DOUBLE DEFAULT 0")
        
        # Ensure api_keys has storage limits (if not already there)
        ensure_column_exists(conn, "api_keys", "storage_limit_mb", "INTEGER DEFAULT 100")
        ensure_column_exists(conn, "api_keys", "request_limit_month", "INTEGER DEFAULT 10000")
        ensure_column_exists(conn, "api_keys", "storage_used_mb", "DOUBLE DEFAULT 0")
        ensure_column_exists(conn, "api_keys", "requests_this_month", "INTEGER DEFAULT 0")

        print("✅ Database initialized and migrated successfully")

# Initialize on import
init_database()
