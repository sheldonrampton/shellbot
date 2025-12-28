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

Shellbot 2.0 will let me find information from all of the places
where I have created content. It will be a "chat-based search
engine" that will not only find relevant content but will summarize
it in response to prompts such as, "What do you think about Julie
Ann Horvath?" or "How do your experiences as a Mormon missionary
relate to your understanding of cognitive biases?" or "Expand
upon your story idea for 'Meeting Mary Again, Ten Years Later."
In addition to responses which summarize my relevant content, it
will provide references back to the original source(s).

In additional to email messages and social media posts,
version 2.0 will include other documents containing Sheldon's
personal knowledge, which may include:
- MS Word, PDF and other documents stored on Sheldon's laptop
- Sheldon's books and other writings
- Google docs
- Google calendar
- Contacts
- Nimble
- Reflect
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
- nuams.com
- nucivic.com
- sheldonrampton@gmail.com Google docs (some NuCivic stuff)
- Mastodon (mas.to)
- racquetmates.net
- PR Watch and Sourcewatch archives (downloadable on prwatch.org)
- Other documents from Sheldon's life
  - Letters
  - Journals
  - Printed writings

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