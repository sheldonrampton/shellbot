#!/usr/bin/env python3
"""
Upload files from laptop to AWS S3 bucket based on filesystem.json configuration.
"""
import os
import json
import sys
from pathlib import Path
from fnmatch import fnmatch
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import hashlib
import time
import datetime
import psycopg2

# S3 bucket name
S3_BUCKET = "shellbot"


def get_db_connection():
    """
    Create a connection to the PostgreSQL database.
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
        print(f"Warning: Could not connect to database: {e}")
        return None


def calculate_file_hash(file_path):
    """
    Calculate SHA256 hash of file for deduplication.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def insert_document_record(conn, file_path):
    """
    Insert a record into laptop_documents table.
    Returns True if successful, False otherwise.
    """
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        
        # Get file metadata
        path = Path(file_path)
        title = path.name
        file_path_str = str(file_path)
        is_mine = True
        last_modified = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        ingested_at = int(time.time())
        document_kind = ""
        priority_rank = 3
        content_hash = calculate_file_hash(file_path)
        
        # Insert record (skip if duplicate hash)
        insert_sql = """
        INSERT INTO public.laptop_documents 
        (title, file_path, is_mine, last_modified_date, ingested_at, document_kind, priority_rank, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_hash) DO NOTHING
        """
        
        cur.execute(insert_sql, (
            title, file_path_str, is_mine, last_modified, 
            ingested_at, document_kind, priority_rank, content_hash
        ))
        
        conn.commit()
        cur.close()
        return True
        
    except Exception as e:
        print(f"  Warning: Could not insert database record: {e}")
        if conn:
            conn.rollback()
        return False


def load_config(config_file="filesystem.json"):
    """Load configuration from filesystem.json"""
    with open(config_file, 'r') as f:
        # Remove comments from JSON (simple approach)
        content = f.read()
        lines = content.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('//')]
        cleaned_content = '\n'.join(filtered_lines)
        return json.loads(cleaned_content)


def expand_path(path):
    """Expand ~ and environment variables in path"""
    return os.path.expanduser(os.path.expandvars(path))


def should_exclude(file_path, exclude_globs):
    """Check if file matches any exclude pattern"""
    # Convert to relative path for matching
    for pattern in exclude_globs:
        if fnmatch(str(file_path), pattern):
            return True
        # Also check if any part of the path matches
        parts = Path(file_path).parts
        for i in range(len(parts)):
            partial_path = str(Path(*parts[i:]))
            if fnmatch(partial_path, pattern.lstrip('**/')) or fnmatch(partial_path, pattern):
                return True
    return False


def has_allowed_extension(file_path, allowed_extensions):
    """Check if file has an allowed extension"""
    if not allowed_extensions:
        return True
    ext = Path(file_path).suffix.lower()
    return ext in [e.lower() for e in allowed_extensions]


def get_files_to_upload(config):
    """Get list of files to upload based on configuration"""
    files_to_upload = []
    excluded_files = []
    large_files = []  # Files larger than 20MB
    
    roots = config.get('roots', [])
    allowed_extensions = config.get('allowedExtensions', [])
    exclude_globs = config.get('excludeGlobs', [])
    max_size_mb = config.get('maxFileSizeMB', 25)
    max_size_bytes = max_size_mb * 1024 * 1024
    large_file_threshold = 20 * 1024 * 1024  # 20MB
    
    for root in roots:
        expanded_root = expand_path(root)
        root_path = Path(expanded_root)
        
        if not root_path.exists():
            print(f"Warning: Root path does not exist: {expanded_root}")
            continue
        
        print(f"Scanning: {expanded_root}")
        
        # Walk through all files in root
        for file_path in root_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    print(f"  Skipping (too large: {file_size / 1024 / 1024:.1f}MB): {file_path}")
                    excluded_files.append((str(file_path), f"Too large ({file_size / 1024 / 1024:.1f}MB)"))
                    continue
                elif file_size > large_file_threshold:
                    large_files.append((str(file_path), file_size))
            except OSError as e:
                print(f"  Skipping (error reading): {file_path} - {e}")
                excluded_files.append((str(file_path), f"Error reading: {e}"))
                continue
            
            # Check if excluded
            if should_exclude(str(file_path), exclude_globs):
                excluded_files.append((str(file_path), "Matched exclude pattern"))
                continue
            
            # Check extension
            if not has_allowed_extension(str(file_path), allowed_extensions):
                excluded_files.append((str(file_path), "Extension not allowed"))
                continue
            
            files_to_upload.append(file_path)
    
    return files_to_upload, excluded_files, large_files


def upload_file_to_s3(local_file, s3_client, bucket_name, s3_key, db_conn=None, max_retries=3):
    """Upload a file to S3 with retry logic and add record to database"""
    for attempt in range(max_retries):
        try:
            s3_client.upload_file(str(local_file), bucket_name, s3_key)
            # If upload succeeded, add database record
            if db_conn:
                insert_document_record(db_conn, local_file)
            return True
        except ClientError as e:
            print(f"  Error uploading {local_file} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False
        except Exception as e:
            # Catch connection errors, timeouts, etc.
            print(f"  Connection error uploading {local_file} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False
            # Wait a bit before retrying
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
    return False


def main():
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: filesystem.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in filesystem.json: {e}")
        sys.exit(1)
    
    # Get list of files to upload
    print("Collecting files to upload...")
    files, excluded_files, large_files = get_files_to_upload(config)
    
    if not files:
        print("No files found matching the criteria")
        return
    
    print(f"\nFound {len(files)} files to upload")
    print(f"Excluded {len(excluded_files)} files")
    print(f"Found {len(large_files)} files larger than 20MB")
    
    # Confirm before uploading
    response = input(f"\nUpload {len(files)} files to S3 bucket '{S3_BUCKET}'? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled")
        return
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
    except NoCredentialsError:
        print("Error: AWS credentials not found. Configure using 'aws configure'")
        sys.exit(1)
    
    # Connect to database
    db_conn = get_db_connection()
    if db_conn:
        print("Connected to database")
    else:
        print("Warning: Database connection failed - will upload to S3 only")
    
    # Upload files
    print("\nUploading files...")
    success_count = 0
    error_count = 0
    failed_uploads = []
    
    for i, file_path in enumerate(files, 1):
        # Create S3 key preserving directory structure
        # Use the path relative to the user's home directory
        home = Path.home()
        try:
            relative_path = file_path.relative_to(home)
            s3_key = str(relative_path)
        except ValueError:
            # File is not under home directory, use absolute path
            s3_key = str(file_path).lstrip('/')
        
        print(f"[{i}/{len(files)}] Uploading: {file_path}")
        print(f"           to S3: {s3_key}")
        
        if upload_file_to_s3(file_path, s3_client, S3_BUCKET, s3_key, db_conn):
            success_count += 1
        else:
            error_count += 1
            failed_uploads.append((str(file_path), s3_key))
    
    print(f"\nUpload complete!")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    
    # Write report
    print(f"\nWriting report to upload_report.txt...")
    with open('upload_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UPLOAD REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Files uploaded: {success_count}\n")
        f.write(f"Upload errors: {error_count}\n")
        f.write(f"Files excluded: {len(excluded_files)}\n")
        f.write(f"Large files (>20MB): {len(large_files)}\n\n")
        
        if failed_uploads:
            f.write("=" * 80 + "\n")
            f.write("FAILED UPLOADS\n")
            f.write("=" * 80 + "\n")
            for file_path, s3_key in failed_uploads:
                f.write(f"{file_path}\n")
                f.write(f"  S3 key: {s3_key}\n\n")
        
        if large_files:
            f.write("=" * 80 + "\n")
            f.write("LARGE FILES (>20MB)\n")
            f.write("=" * 80 + "\n")
            # Sort by size, largest first
            large_files.sort(key=lambda x: x[1], reverse=True)
            for file_path, size in large_files:
                f.write(f"{size / 1024 / 1024:>8.2f} MB  {file_path}\n")
            f.write("\n")
        
        if excluded_files:
            f.write("=" * 80 + "\n")
            f.write("EXCLUDED FILES\n")
            f.write("=" * 80 + "\n")
            for file_path, reason in excluded_files:
                f.write(f"{reason:30s}  {file_path}\n")
    
    print(f"Report written to upload_report.txt")
    
    # Close database connection
    if db_conn:
        db_conn.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
