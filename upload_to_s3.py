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

# S3 bucket name
S3_BUCKET = "shellbot"


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


def upload_file_to_s3(local_file, s3_client, bucket_name, s3_key):
    """Upload a file to S3"""
    try:
        s3_client.upload_file(str(local_file), bucket_name, s3_key)
        return True
    except ClientError as e:
        print(f"  Error uploading {local_file}: {e}")
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
    
    # Upload files
    print("\nUploading files...")
    success_count = 0
    error_count = 0
    
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
        
        if upload_file_to_s3(file_path, s3_client, S3_BUCKET, s3_key):
            success_count += 1
        else:
            error_count += 1
    
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


if __name__ == "__main__":
    main()
