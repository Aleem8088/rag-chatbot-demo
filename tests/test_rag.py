import io

import pytest

from app.chunking import Chunk, chunk_file_bytes, chunk_text, load_directory
from app.main import DOCS_DIR, create_app
from app.rag import NOT_FOUND_MESSAGE, answer_question
from app.retriever import Retriever


class FakeLLM:
    """Returns a canned reply and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(load_directory(DOCS_DIR))


def test_chunking_keeps_heading_as_location():
    chunks = chunk_text("a.md", "# Title\n\n## Returns\n\nYou have 30 days.\n\n## Shipping\n\nTakes 3 days.")
    assert [c.location for c in chunks] == ["section: Returns", "section: Shipping"]
    assert chunks[0].text == "You have 30 days."


def test_long_paragraph_is_split():
    text = "## A\n\n" + " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_text("a.md", text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 760 for c in chunks)


def test_unsupported_file_type_rejected():
    with pytest.raises(ValueError):
        chunk_file_bytes("evil.exe", b"x")


def test_retriever_finds_the_right_document(retriever):
    top, _ = retriever.search("How much does express shipping cost?", k=1)[0]
    assert top.source == "shipping-faq.md"
    assert "$15" in top.text


def test_retriever_returns_nothing_for_gibberish(retriever):
    assert retriever.search("zzzz qqqq xxxx", k=4) == []


def test_answer_cites_only_real_passages(retriever):
    llm = FakeLLM("Express shipping costs $15 [S1]. Also see [S99].")
    result = answer_question("How much does express shipping cost?", retriever, llm)
    assert result.found
    assert [c.label for c in result.citations] == ["S1"]  # S99 does not exist and is dropped
    assert result.citations[0].source == "shipping-faq.md"


def test_not_found_from_model_is_reported(retriever):
    result = answer_question("How long is the warranty on tents?", retriever, FakeLLM("NOT_FOUND"))
    assert not result.found
    assert result.text == NOT_FOUND_MESSAGE


def test_out_of_scope_question_never_calls_the_model(retriever):
    llm = FakeLLM("should not be used")
    result = answer_question("What is the CEO's name?", retriever, llm)
    assert not result.found
    assert llm.calls == []


def test_prompt_treats_context_as_data(retriever):
    llm = FakeLLM("30 days [S1]")
    answer_question("How many days do I have to return an item?", retriever, llm)
    system, user = llm.calls[0]
    assert "not instructions" in system
    assert "[S1]" in user and "Question:" in user


@pytest.fixture()
def client():
    app = create_app(llm=FakeLLM("Returns take 30 days [S1]."))
    return app.test_client()


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_ask_requires_a_question(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 400


def test_ask_rejects_very_long_questions(client):
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 400


def test_ask_returns_citations(client):
    body = client.post("/ask", json={"question": "How many days do I have to return an item?"}).get_json()
    assert body["found"] is True
    assert body["citations"] and body["citations"][0]["source"] == "returns-policy.md"


def test_upload_adds_a_searchable_document(client):
    data = {"file": (io.BytesIO(b"# Gift wrap\n\nGift wrapping costs $4 per parcel."), "gift.md")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = client.post("/ask", json={"question": "How much is gift wrapping?"}).get_json()
    assert body["found"] is True


def test_upload_rejects_bad_extension(client):
    data = {"file": (io.BytesIO(b"x"), "run.sh")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_chunk_dataclass_is_hashable():
    assert hash(Chunk("a", "", "b"))
