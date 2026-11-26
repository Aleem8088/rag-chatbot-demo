# Document Q&A with Citations (RAG chatbot demo)

A small chatbot that answers questions **only from your documents**, cites the passage each claim came from, and says "I couldn't find that" instead of guessing.

> Built from scratch as a portfolio demo. The sample documents are for a fictional store, so there is no real or client data in this repo.

## What it does
- Loads `.md`, `.txt` and `.pdf` files, splits them into passages that remember their source (file + page or section).
- Retrieves the best passages for a question (BM25 keyword search).
- Sends only those passages to Claude with strict rules: answer from the passages, cite as `[S1]`, say `NOT_FOUND` otherwise, and treat passage text as data, not instructions.
- Shows the answer with the cited passages, so a reader can check every claim.
- Lets you upload more documents in the browser.

## Why these choices
- **Citations are validated.** The app drops any citation number that doesn't map to a passage that was actually retrieved.
- **Out-of-scope questions never reach the model.** If retrieval finds nothing relevant, the app answers "not found" without an API call: cheaper and safer.
- **BM25 instead of embeddings.** It needs no model download and runs anywhere. The `Retriever` is a small class, so embeddings or hybrid search can replace it for a real project (better for paraphrased questions).
- **No secrets in code.** The API key comes from the `ANTHROPIC_API_KEY` environment variable.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY="..."
gunicorn -b 127.0.0.1:8000 app.main:app     # or: flask --app app.main run
```
Open http://127.0.0.1:8000. Each question makes one API call, so it uses your own API credits.

With Docker: `docker compose up --build` (reads `ANTHROPIC_API_KEY` from your environment or a `.env` file).

The default model is `claude-opus-5-5`. Set `ANTHROPIC_MODEL` to use a different one, and `ANTHROPIC_EFFORT` (default `low`) to trade depth for cost.

## Tests and evaluation

```bash
ruff check . && pytest -q       # 16 tests, no API key needed (the model is faked)
python -m eval.run_eval         # retrieval eval: free, offline, runs in CI
python -m eval.run_eval --llm   # answer eval: uses the API (about 10 calls)
```

- **Retrieval eval:** for each answerable question, is the right passage in the top results?
- **Answer eval (`--llm`):** does the bot answer with a citation, and does it decline the unanswerable questions? Keyword search can match a stray word ("gift" in "gift cards"), so declining is verified here, not in the retrieval eval.

## Status of this demo
- ✅ Lint, 16 unit tests and the retrieval eval pass locally.
- ⏳ **Not yet run against the live API**: the answer eval (`--llm`) and a real browser session haven't been executed. Run them with your own key before relying on the results.
- ⏳ The Docker image build hasn't been verified locally; CI runs `docker build` on every push.

## Limits (honest list)
- Keyword retrieval misses paraphrases ("send it back" vs "return"). Use embeddings for that.
- Uploaded documents live in memory and disappear on restart.
- No login. Don't expose it publicly with sensitive documents.
- Scanned PDFs (images) have no extractable text.

## What I'd adapt for your project
Your documents and file types, a database or vector store, authentication, a chat widget for your site, WhatsApp or Slack delivery, and an evaluation set built from your real questions.
