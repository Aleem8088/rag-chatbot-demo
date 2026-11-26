# Loom Script: "Chatbot that cites its sources" (about 2 minutes)

Record only after you've run it with your own API key and the answers look right.

## Before recording
- [ ] `docker compose up` (or gunicorn) running, page open at http://127.0.0.1:8000
- [ ] Run `python -m eval.run_eval --llm` once and keep the result handy
- [ ] Close notifications; use a clean browser window

## Script

**0:00 Intro (15s)**
"Hi, I'm Mohammed. This is a chatbot that answers only from your own documents, shows where each answer came from, and says so when it doesn't know."

**0:15 A normal question (30s)**
Ask: "How many days do I have to return an item?"
"The answer comes with the exact passages it used. I can check every claim."

**0:45 A question it can't answer (25s)**
Ask: "What is the CEO's name?"
"This isn't in the documents, so it says it couldn't find it instead of making something up."

**1:10 Upload a document (25s)**
Upload a small text file, then ask about it.
"New documents are searchable immediately."

**1:35 How I know it works (20s)**
Show the terminal with the eval output.
"I test it with a set of questions, including ones that should be refused. For your project I'd build this from your real questions."

**1:55 Close (10s)**
"The code is linked below. If you want this on your own documents, send me a message."

## Tips
- Don't show your API key or `.env` file on screen.
- Add captions. Many viewers watch muted.
