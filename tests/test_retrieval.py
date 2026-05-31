"""Module 8 Lab autograder — runs against a real Weaviate service in CI.

Tests follow Section G of the build packet:
- Structural: module imports, required functions present
- Schema correctness: Post class, vectorizer none, multi-property BM25
- Ingest correctness: full-corpus line-count match
- Retriever shape: list[str] of length <= k
- Behavioral: recall@5 floors on a 5-pair fixture (BM25 >= 0.6, dense >= 0.6,
  hybrid >= 0.7)
- evaluate_retriever output shape

The autograder reads `data/corpus.jsonl` to derive the expected ingest count,
which is robust to corpus rebuilds.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import weaviate
from sentence_transformers import SentenceTransformer

import retrieval as retrieval_mod
from retrieval import (
    bm25_search,
    create_schema,
    dense_search,
    evaluate_retriever,
    hybrid_search,
    index_corpus,
)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "retrieval_eval.jsonl")
CLASS_NAME = "Post"
REQUIRED_BM25_PROPS = {"title", "question_text", "answer_text"}


def _wait_for_weaviate(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if weaviate.Client(url).is_ready():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Weaviate not ready at {url} within {timeout}s")


@pytest.fixture(scope="session")
def client() -> weaviate.Client:
    _wait_for_weaviate(WEAVIATE_URL)
    return weaviate.Client(WEAVIATE_URL)


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def expected_corpus_count() -> int:
    with open(CORPUS_PATH) as f:
        return sum(1 for _ in f)


@pytest.fixture(scope="session")
def ingested(client, embedder):
    """Run create_schema + index_corpus once for the whole session."""
    create_schema(client)
    count = index_corpus(client, CORPUS_PATH, embedder)
    return count


@pytest.fixture(scope="session")
def fixture_subset() -> list[dict]:
    """Use the first 5 rows of the labeled eval as the behavioral fixture.

    The behavioral floors (BM25 >= 0.6, dense >= 0.6, hybrid >= 0.7) are
    calibrated against the answer-key implementation on this 5-pair subset.
    """
    rows = []
    with open(EVAL_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == 5:
                break
    return rows


def _recall_at_5(rows: list[dict], search_fn) -> float:
    hits = 0
    for row in rows:
        top = search_fn(row["query"], 10)
        if row["gold_doc_id"] in top[:5]:
            hits += 1
    return hits / len(rows) if rows else 0.0


def test_retrieval_module_imports():
    for name in (
        "create_schema",
        "index_corpus",
        "bm25_search",
        "dense_search",
        "hybrid_search",
        "evaluate_retriever",
    ):
        assert callable(getattr(retrieval_mod, name, None)), f"{name} must be defined and callable"


def test_create_schema_creates_post_class_with_vectorizer_none(client, ingested):
    assert client.schema.exists(CLASS_NAME), "Post class must exist after create_schema"
    schema = client.schema.get(CLASS_NAME)
    assert schema["vectorizer"] == "none", f"vectorizer must be 'none', got {schema['vectorizer']!r}"
    prop_names = {p["name"] for p in schema["properties"]}
    for required in ("doc_id", "subset", "title", "question_text", "answer_text", "text"):
        assert required in prop_names, f"property {required!r} missing from Post schema"
    bm25_indexed = {
        p["name"] for p in schema["properties"]
        if p.get("indexInverted", True) is not False
        and p.get("tokenization") in (None, "word", "lowercase", "whitespace", "field")
    }
    for required in REQUIRED_BM25_PROPS:
        assert required in bm25_indexed, f"{required!r} must be BM25-indexed (inverted index enabled)"


def test_index_corpus_ingests_full_corpus(client, ingested, expected_corpus_count):
    assert ingested == expected_corpus_count, (
        f"index_corpus return value {ingested} != corpus line count {expected_corpus_count}"
    )
    agg = (
        client.query.aggregate(CLASS_NAME)
        .with_meta_count()
        .do()
    )
    actual = agg["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
    assert actual == expected_corpus_count, (
        f"Aggregate count {actual} != corpus line count {expected_corpus_count}"
    )


def test_each_search_returns_list_of_str_at_most_k(client, embedder, ingested):
    q = "how do I rebase a feature branch"
    for name, fn in (
        ("bm25_search", lambda: bm25_search(client, q, 5)),
        ("dense_search", lambda: dense_search(client, q, 5, embedder)),
        ("hybrid_search", lambda: hybrid_search(client, q, 5, embedder, 0.5)),
    ):
        result = fn()
        assert isinstance(result, list), f"{name} must return a list, got {type(result).__name__}"
        assert all(isinstance(x, str) for x in result), f"{name} items must be str"
        assert len(result) <= 5, f"{name} returned {len(result)} > k=5"


def test_bm25_recall_at_5_floor(client, ingested, fixture_subset):
    score = _recall_at_5(fixture_subset, lambda q, k: bm25_search(client, q, k))
    assert score >= 0.6, f"BM25 recall@5 on fixture = {score}, floor = 0.6"


def test_dense_recall_at_5_floor(client, embedder, ingested, fixture_subset):
    score = _recall_at_5(fixture_subset, lambda q, k: dense_search(client, q, k, embedder))
    assert score >= 0.6, f"dense recall@5 on fixture = {score}, floor = 0.6"


def test_hybrid_recall_at_5_floor(client, embedder, ingested, fixture_subset):
    score = _recall_at_5(fixture_subset, lambda q, k: hybrid_search(client, q, k, embedder, 0.5))
    assert score >= 0.7, f"hybrid recall@5 on fixture = {score}, floor = 0.7"


def test_evaluate_retriever_returns_expected_keys(client, ingested):
    out = evaluate_retriever(EVAL_PATH, lambda q, k: bm25_search(client, q, k))
    assert isinstance(out, dict)
    for key in ("recall@5", "recall@10", "mrr"):
        assert key in out, f"missing key {key!r}"
        v = out[key]
        assert isinstance(v, (int, float)), f"{key} must be numeric, got {type(v).__name__}"
        assert 0.0 <= float(v) <= 1.0, f"{key}={v} out of [0, 1]"


def test_comparison_brief_has_substance():
    import re
    brief_path = os.path.join(os.path.dirname(__file__), "..", "comparison_brief.md")
    assert os.path.exists(brief_path), "comparison_brief.md missing"
    body = open(brief_path).read()
    assert len(body) >= 250, f"comparison_brief.md too short ({len(body)} chars; need >= 250)"
    lower = body.lower()
    for needle in ("bm25", "dense", "hybrid"):
        assert needle in lower, f"comparison_brief.md must mention {needle!r}"
    for placeholder in ("_your number_", "_query 1 — explanation_", "_query 2 — explanation_"):
        assert placeholder not in body, (
            f"comparison_brief.md still contains template placeholder {placeholder!r}; "
            "replace placeholders with your analysis"
        )
    metric_numbers = re.findall(r"\d+\.\d+", body)
    assert len(metric_numbers) >= 15, (
        f"comparison_brief.md metrics table looks unfilled: found {len(metric_numbers)} decimal "
        "numbers; need at least 15 (3 retrievers × recall@5/recall@10/MRR/factoid recall@5/paraphrastic recall@5)"
    )
