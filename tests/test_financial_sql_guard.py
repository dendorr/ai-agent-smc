"""Smoke tests for the model-generated SQL guard in scripts/financial_agent.py.

These tests do not call Ollama or touch the real ChromaDB: the persistent client
and collection opening are patched during import, and every test runs against a
temporary SQLite file.
"""

import asyncio
import os
import sqlite3
import sys
import time
from unittest import mock

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

with mock.patch("chromadb.PersistentClient"), mock.patch("embeddings.open_collection"):
    import financial_agent as fa  # noqa: E402


ROW_COUNT = 1000


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "financial_files.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE "f_sales_0001" (client TEXT, amount REAL, day TEXT)')
    conn.execute('CREATE TABLE "f_other_0002" (client TEXT, amount REAL)')
    conn.execute('CREATE TABLE "f_hidden_0003" (secret TEXT)')
    conn.execute("CREATE TABLE _file_schemas (table_name TEXT)")
    conn.executemany(
        "INSERT INTO _file_schemas VALUES (?)", [("f_sales_0001",), ("f_other_0002",)]
    )
    conn.executemany(
        'INSERT INTO "f_sales_0001" VALUES (?, ?, ?)',
        [(f"C{i % 7}", float(i), "2026-01-01") for i in range(ROW_COUNT)],
    )
    conn.executemany('INSERT INTO "f_other_0002" VALUES (?, ?)', [("X", 1.0), ("Y", 2.0)])
    conn.execute("INSERT INTO \"f_hidden_0003\" VALUES ('do-not-read')")
    # A view registered as indexed that reads a non-indexed table
    conn.execute('CREATE VIEW "f_view_0004" AS SELECT secret FROM "f_hidden_0003"')
    conn.execute("INSERT INTO _file_schemas VALUES ('f_view_0004')")
    conn.commit()
    conn.close()
    return path


def _ok(rows):
    return bool(rows) and "error" not in rows[0]


def _run(path, sql):
    rows, _ = fa.query_db_generated(path, sql)
    return rows


def _table_count(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
    finally:
        conn.close()


VALID_QUERIES = [
    'SELECT client, SUM(amount) AS total FROM "f_sales_0001" GROUP BY client '
    "ORDER BY total DESC LIMIT 30",
    "SELECT * FROM F_SALES_0001 WHERE client LIKE '%c1%' LIMIT 5",
    "SELECT strftime('%Y', day), count(*) FROM f_sales_0001 GROUP BY 1 LIMIT 5",
    "WITH t AS (SELECT * FROM f_sales_0001) SELECT client FROM t LIMIT 5",
    "SELECT client FROM f_sales_0001 UNION SELECT client FROM f_other_0002",
    "SELECT client FROM f_sales_0001 EXCEPT SELECT client FROM f_other_0002",
    "SELECT hex(client) FROM f_sales_0001 LIMIT 1;",
    "SELECT * FROM f_sales_0001 LIMIT 3; -- trailing comment",
]


@pytest.mark.parametrize("sql", VALID_QUERIES)
def test_valid_queries_run(db_path, sql):
    assert _ok(_run(db_path, sql))


DANGEROUS_QUERIES = [
    'DROP TABLE "f_sales_0001"',
    'DELETE FROM "f_sales_0001"',
    "INSERT INTO f_sales_0001 VALUES ('x', 1, 'd')",
    "UPDATE f_sales_0001 SET amount = 0",
    "CREATE TABLE z (a)",
    "ATTACH DATABASE ':memory:' AS other",
    "PRAGMA table_info(f_sales_0001)",
    "PRAGMA journal_mode = WAL",
    "SELECT 1; DROP TABLE f_sales_0001",
    "SELECT * FROM f_sales_0001; /* c */ DROP TABLE f_sales_0001",
    "SELECT * FROM _file_schemas",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM SQLITE_SCHEMA",
    "SELECT * FROM main.f_sales_0001",
    "SELECT * FROM pragma_table_info('f_sales_0001')",
    "SELECT * FROM f_hidden_0003",
    "SELECT client FROM f_sales_0001 UNION SELECT secret FROM f_hidden_0003",
    "SELECT client FROM f_sales_0001 UNION SELECT table_name FROM _file_schemas",
    "SELECT * FROM f_sales_0001 WHERE client IN (SELECT secret FROM f_hidden_0003)",
    "SELECT load_extension('x')",
    "SELECT randomblob(1000000000)",
    "SELECT printf('%.*c', 100000000, 'x')",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r",
    "not sql at all ;;",
]


@pytest.mark.parametrize("sql", DANGEROUS_QUERIES)
def test_dangerous_queries_blocked(db_path, sql):
    tables_before = _table_count(db_path)
    rows = _run(db_path, sql)
    assert rows and "error" in rows[0]
    assert _table_count(db_path) == tables_before


@pytest.mark.parametrize("limit_clause", ["", "LIMIT 100000", "LIMIT -1", "LIMIT 5, 100000",
                                          "LIMIT (SELECT 100000)"])
def test_limit_is_added_or_capped(db_path, limit_clause):
    rows = _run(db_path, f"SELECT * FROM f_sales_0001 {limit_clause}")
    assert _ok(rows)
    assert len(rows) == fa.GENERATED_SQL_MAX_LIMIT


def test_lower_limit_is_kept(db_path):
    assert len(_run(db_path, "SELECT * FROM f_sales_0001 LIMIT 7")) == 7


def test_union_limit_applies_to_outer_result(db_path):
    sql = "SELECT amount FROM f_sales_0001 UNION ALL SELECT amount FROM f_other_0002"
    assert len(_run(db_path, sql)) == fa.GENERATED_SQL_MAX_LIMIT


def test_slow_query_times_out(db_path, monkeypatch):
    monkeypatch.setattr(fa, "GENERATED_SQL_TIMEOUT_S", 0.3)
    start = time.monotonic()
    rows = _run(
        db_path, "SELECT count(*) FROM f_sales_0001 a, f_sales_0001 b, f_sales_0001 c"
    )
    assert rows == [{"error": "interrupted"}]
    assert time.monotonic() - start < 3


def test_internal_query_db_still_reads_schemas(db_path):
    rows = fa.query_db(db_path, "SELECT table_name FROM _file_schemas")
    assert {r["table_name"] for r in rows} == {"f_sales_0001", "f_other_0002", "f_view_0004"}


def test_internal_query_db_is_read_only(db_path):
    tables_before = _table_count(db_path)
    rows = fa.query_db(db_path, 'DROP TABLE "f_sales_0001"')
    assert rows and "error" in rows[0]
    assert _table_count(db_path) == tables_before


def test_deep_nesting_rejected_without_exception(db_path):
    sql = "SELECT " + "(" * 3000 + "1" + ")" * 3000 + " FROM f_sales_0001"
    assert len(sql) <= fa.GENERATED_SQL_MAX_CHARS
    rows, normalized = fa.query_db_generated(db_path, sql)
    assert rows and "RecursionError" in rows[0]["error"]
    assert normalized is None


def test_deep_union_rejected_without_exception(db_path, monkeypatch):
    monkeypatch.setattr(fa, "GENERATED_SQL_MAX_CHARS", 1_000_000)
    sql = " UNION ".join(["SELECT client FROM f_sales_0001"] * 3000)
    rows, normalized = fa.query_db_generated(db_path, sql)
    assert rows and "RecursionError" in rows[0]["error"]
    assert normalized is None


def test_overlong_sql_rejected(db_path):
    sql = "SELECT client FROM f_sales_0001 WHERE client = '" + "x" * 10_001 + "'"
    rows, normalized = fa.query_db_generated(db_path, sql)
    assert rows and "too long" in rows[0]["error"]
    assert normalized is None


def test_huge_group_concat_hits_length_limit(db_path):
    rows = _run(
        db_path,
        "SELECT length(group_concat(hex(a.day) || hex(b.day))) "
        "FROM f_sales_0001 a, f_sales_0001 b",
    )
    assert rows and "error" in rows[0]
    assert "too big" in rows[0]["error"]


def test_view_cannot_reach_non_indexed_table(db_path):
    rows = _run(db_path, 'SELECT * FROM "f_view_0004"')
    assert rows and "error" in rows[0]
    assert "f_hidden_0003" in rows[0]["error"]


def test_authorizer_denies_read_outside_allowed_set():
    auth = fa._make_generated_sql_authorizer({"f_sales_0001"})
    assert auth(fa.sqlite3.SQLITE_READ, "f_sales_0001", "client", "main", None) == fa.sqlite3.SQLITE_OK
    assert auth(fa.sqlite3.SQLITE_READ, "f_hidden_0003", "secret", "main", None) == fa.sqlite3.SQLITE_DENY
    assert auth(fa.sqlite3.SQLITE_READ, "_file_schemas", "table_name", "main", None) == fa.sqlite3.SQLITE_DENY


def test_returns_normalized_sql(db_path):
    rows, normalized = fa.query_db_generated(
        db_path, "SELECT client FROM f_sales_0001 /* note */ LIMIT 99999"
    )
    assert _ok(rows)
    assert normalized == f"SELECT client FROM f_sales_0001 LIMIT {fa.GENERATED_SQL_MAX_LIMIT}"


def test_adaptive_context_shows_normalized_sql(db_path, monkeypatch):
    async def fake_route_query(query, schemas):
        return [{"table": "f_sales_0001", "sql": "SELECT client FROM f_sales_0001 -- c"}]

    monkeypatch.setattr(fa, "route_query", fake_route_query)
    monkeypatch.setattr(fa, "GENERIC_DB", db_path)
    schemas = [{"table_name": "f_sales_0001", "filename": "sales.xlsx", "sheet": "S1",
                "row_count": ROW_COUNT}]
    context = asyncio.run(fa.adaptive_sql_context("q", schemas))
    assert f"SQL: SELECT client FROM f_sales_0001 LIMIT {fa.GENERATED_SQL_MAX_LIMIT}" in context
    assert "-- c" not in context
