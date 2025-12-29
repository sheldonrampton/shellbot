#!/usr/bin/env python3
"""
Check the laptop_documents table to see how many records have been inserted.
"""
import psycopg2

db_config = {
    'dbname': 'd4hob51sunu43p',
    'user': 'u9ghihqtsvipjj',
    'password': 'pc39c0e273101856fa90ead0a6c98f774641fb4933a3e1a34fa901f91350a5bb2',
    'host': 'ccba8a0vn4fb2p.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com',
    'port': '5432',
    'sslmode': 'allow'
}

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# Count total records
cur.execute("SELECT COUNT(*) FROM public.laptop_documents")
count = cur.fetchone()[0]
print(f"Total records in laptop_documents: {count}")

# Show most recent 5 records
cur.execute("""
    SELECT title, file_path, last_modified_date, ingested_at 
    FROM public.laptop_documents 
    ORDER BY ingested_at DESC 
    LIMIT 5
""")

print("\nMost recent 5 uploads:")
print("-" * 100)
for row in cur.fetchall():
    title, file_path, last_modified, ingested_at = row
    print(f"{title}")
    print(f"  Path: {file_path}")
    print(f"  Modified: {last_modified}")
    print(f"  Ingested: {ingested_at}")
    print()

cur.close()
conn.close()
