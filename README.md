# Shellbot

Shellbot 2.0 is an LLM-based personal knowledge curator, personal assistant, and blogging/social media copilot.

## Overview

Shellbot 1.0 was a RAG chatbot built using the OpenAI API that
answered prompts based on a knowledge base containing:
- Social media posts (Facebook, Twitter)
- Email messages
- Other writings by Sheldon Rampton

Shellbot 2.0 builds on Shellbot 1.0 to provide the following
functionality:

### Personal knowledge curator

Goal: unify scattered info, preserve provenance, and rank it by recency + “current-belief alignment.”

In additional to email messages and social media posts,
version 2.0 will include other documents containing Sheldon's
personal knowledge, which may include:
- MS Word, PDF and other documents stored on Sheldon's laptop
- Google docs
- Google calendar
- Contacts
- Nimble
- Code repositories on Github
- Strong Silent Type blog posts
- Anki cards
- ChatGPT conversations
- DevonThink
- EverNote
- YouTube videos
- Text messages
- Notes
- Slack
- VoodooPad
- Things
- CoCoDems correspondence
- LinkedIn
- Wikipedia
- GEM Wiki
- MarchantFamily.com
- HapSmith.com
- del.icio.us
- disqus.com
- nextdoor.com
- Mastodon (mas.to)
- racquetmates.net

Core design choices:

Canonical document model: every item gets {source, source_id, created_at, ingested_at, author, title, url/path, permissions, content_hash}

Provenance-first: every answer should carry links/IDs back to originals

Versioning: store multiple snapshots rather than overwriting

How to handle “my views changed”

Segment by time windows (“Sheldon-2005”, “Sheldon-2015”, “Sheldon-2025”)

Compute embeddings per segment; retrieve from the right era on purpose

Add a lightweight “belief delta” tag: “this view appears to be earlier/older than your recent writing” (you can implement this as a ranking feature rather than pretending you can perfectly infer beliefs)

### Personal assistant (email + calendar + YouTube)

Start with assistive automation rather than full autonomy:

triage, summarize, draft replies, propose deletes/labels, propose calendar blocks

then add execution behind the two-step pattern

### Blogging + social media copilot

Once the curator exists, it becomes my “research brain”:

“Turn yesterday’s Reflect entry + related past posts into a noir-style draft”

“Generate 5 platform-specific variants + schedule suggestions”

“Track performance + learn what tone/topics get engagement”

Post blog-related announcements to social media; include links to blog posts in social media comments.