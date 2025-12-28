# Shellbot Schema

Shellbot 2.0 is an LLM-based personal knowledge curator, personal assistant, and blogging/social media copilot.

## Shellbot 1.0 schema

See shellbot1_schema.sql

### entries
Captures a log of user interactions with the chatbot. (This table is not
needed for Shellbot 2.0.)
- session_id: text
- entry_timestamp: text
- user_input text
- bot_response text

### gmail_messages
Copies of email messages from gMail
- id: integer (a unique ID)
- subject: text (the email subject line)
- timestamp: integer (when the email was sent)
- from_email: text (the sender)
- to_emails: text (the recipients)
- message: text (the body of the email)

### social_posts
Social media posts from Twitter and Facebook
- id: integer (a unique ID)
- platform: text ("Tweet" or "Facebook post" or "Facebook comment")
- platform_id: text (the platform's unique ID for the content)
- timestamp: text (when the item was posted)
- content: text (The text of the post)
- url: text (a primary URL that was included in the post)

### shellbot_knowledge
Chunked content from social media and emails
- vector_id: text (the Pinecone vector ID of the chunk)
- platform: text ("Email", "Tweet", "Facebook post", "Facebook comment")
- title: text (the date, platform, and beginning of the content)
- unix_timestamp: integer (When the item was posted)
- formatted_datetime: text (When the item was posted, formatted for display)
- content: text (the full chunk)
- url: text (Some tweets and Facebook posts/comments have primary URLs; for emails, the "url" consists of the from and to addresses, comma-separated.)

## Ideas for Shellbot 2.0 schema

source, source_id, created_at, ingested_at, author, title, url/path, permissions, content_hash

### Core tables

#### sources

- source_id (pk)
- type (filesystem, gdrive, gmail, etc.)
- display_name
- config_json (encrypted-at-rest if possible)
- auth_ref (pointer to stored credentials)
- change_cursor (Drive token, Gmail historyId, etc.)
- enabled, last_sync_at, status

#### documents (the logical “thing”)

- document_id (pk)
- source_id (fk)
- source_doc_key (unique per source: Drive fileId, filepath, gmail messageId/threadId)
- doc_type (email, doc, pdf, markdown, tweet, etc.)
- title
- created_at_source, updated_at_source
- canonical_url (Drive link, Gmail link, file:// or your own file viewer link)
- deleted_at_source (for change detection)


#### CanonicalDocument (alternate structure)

- source (e.g., laptop_fs, gdrive_doc)
- source_id (stable ID from that system)
- created_at (from source if possible)
- ingested_at
- author
- title
- url_or_path
- permissions (even if “local only” for now)
- content_hash
- plus: content_text (or pointer), metadata_json

#### document_versions (the content over time)

- version_id (pk)
- document_id (fk)
- version_label (optional; Drive revision ID if you want)
- content_text (or store in object storage + keep pointer)
- content_hash (for dedupe/change detect)
- extracted_metadata_json (author, recipients, etc.)
- observed_at (when you ingested it)

#### chunks

- chunk_id (pk)
- version_id (fk)
- chunk_index
- chunk_text
- chunk_hash
- embedding_ref (if embeddings stored elsewhere)
- token_count

#### entities / tags / document_tags (optional now, useful soon)

For topics (“climate”, “media criticism”), people, orgs, etc.

### Why this structure helps you later

Incremental ingestion is natural: new versions only when content_hash changes.

Timeline-aware answers are possible because you can query versions by date.

Multi-source is normal; you’re not hardcoding “email-ness” into everything.