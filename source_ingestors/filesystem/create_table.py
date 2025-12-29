#!/usr/bin/env python3
"""
Script to create the laptop_documents table in PostgreSQL database.
"""

import os
import sys
import psycopg2
from psycopg2 import sql


def get_db_connection():
    """
    Create a connection to the PostgreSQL database using credentials from mcp.json.
    """
    db_config = {
        'dbname': 'd4hob51sunu43p',
        'user': 'u9ghihqtsvipjj',
        'password': 'pc39c0e273101856fa90ead0a6c98f774641fb4933a3e1a34fa901f91350a5bb2',
        'host': 'ccba8a0vn4fb2p.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com',
        'port': '5432',
        'sslmode': 'allow'
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def create_laptop_documents_table(conn):
    """
    Create the laptop_documents table if it doesn't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS public.laptop_documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        file_path TEXT,
        is_mine BOOLEAN,
        last_modified_date TEXT,
        ingested_at INTEGER,
        document_kind TEXT,
        priority_rank INTEGER,
        content_hash TEXT UNIQUE
    );
    """
    
    # Create index on content_hash for efficient duplicate detection
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_laptop_documents_content_hash 
    ON public.laptop_documents(content_hash);
    """
    
    # Create index on file_path for efficient lookups
    create_path_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_laptop_documents_file_path 
    ON public.laptop_documents(file_path);
    """
    
    try:
        cur = conn.cursor()
        
        print("Creating laptop_documents table...")
        cur.execute(create_table_sql)
        
        print("Creating indexes...")
        cur.execute(create_index_sql)
        cur.execute(create_path_index_sql)
        
        conn.commit()
        
        print("\n✓ Table 'laptop_documents' created successfully")
        print("✓ Indexes created successfully")
        
        # Get table info
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'laptop_documents'
            ORDER BY ordinal_position;
        """)
        
        columns = cur.fetchall()
        print("\nTable structure:")
        print("-" * 60)
        for col_name, data_type, nullable in columns:
            null_str = "NULL" if nullable == "YES" else "NOT NULL"
            print(f"  {col_name:20s} {data_type:15s} {null_str}")
        print("-" * 60)
        
        cur.close()
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating table: {e}")
        sys.exit(1)


def main():
    """
    Main function to create the laptop_documents table.
    """
    print("=" * 60)
    print("Creating laptop_documents table in PostgreSQL")
    print("=" * 60)
    print()
    
    # Get database connection
    conn = get_db_connection()
    
    # Create the table
    create_laptop_documents_table(conn)
    
    # Close connection
    conn.close()
    print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
