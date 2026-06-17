"""
SQLite persistence layer for the Nifty 100 Swing Trading Agent.

Schema:
  - scans:   one row per calendar date a scan was run. Re-running the
             pipeline on the same date OVERWRITES the previous row, which
             is how duplicate same-day alerts are avoided.
  - signals: the shortlisted stocks belonging to a given scan_date.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import libsql_client
import pandas as pd
import streamlit as st

import config

# Raised by either backend on a query/connection failure.
_DB_ERRORS = (sqlite3.Error, libsql_client.LibsqlError)


class _Row:
    """Wraps a libsql_client row + its column names so it behaves like a
    sqlite3.Row - supports row["col"], row[0], and dict(row)."""

    __slots__ = ("_columns", "_values")

    def __init__(self, columns, values):
        self._columns = columns
        self._values = values

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._columns.index(key)]
        return self._values[key]

    def keys(self):
        return self._columns


class _CursorResult:
    """Mimics the subset of sqlite3's cursor interface used in this module."""

    def __init__(self, result_set):
        self._columns = list(result_set.columns)
        self._rows = result_set.rows

    def __iter__(self):
        return (_Row(self._columns, r) for r in self._rows)

    def fetchone(self):
        if not self._rows:
            return None
        return _Row(self._columns, self._rows[0])

    def fetchall(self):
        return [_Row(self._columns, r) for r in self._rows]


_turso_client = None


def _get_turso_client(url, auth_token):
    """Returns a process-wide libsql_client, created once and reused for
    every get_connection() call - creating a new client per query spins up
    a new thread, event loop and HTTP session each time, which made every
    page interaction noticeably slower once the app moved to Turso."""
    global _turso_client
    if _turso_client is None:
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        _turso_client = libsql_client.create_client_sync(url, auth_token=auth_token)
    return _turso_client


class _TursoConnection:
    """Adapts the shared libsql_client HTTP client to the connection methods
    (execute/commit/rollback/close) used by get_connection()'s callers.

    Uses the HTTP-based Hrana protocol (https://) rather than the
    WebSocket-based one (libsql:// / wss://): Streamlit Community Cloud's
    network breaks the WebSocket upgrade, causing a WSServerHandshakeError.
    The HTTP client has no transactions, so each statement commits
    immediately and commit()/rollback() are no-ops. The underlying client is
    a process-wide singleton (see _get_turso_client), so close() is a no-op.
    """

    def __init__(self, url, auth_token):
        self._client = _get_turso_client(url, auth_token)

    def execute(self, sql, params=()):
        return _CursorResult(self._client.execute(sql, list(params)))

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@contextmanager
def get_connection():
    """Yields a connection, committing on success and rolling back on error.

    Uses a hosted Turso (libSQL) database when TURSO_DATABASE_URL /
    TURSO_AUTH_TOKEN are configured - so data survives Streamlit Cloud
    restarts/redeploys - otherwise falls back to the local SQLite file.
    """
    if config.TURSO_DATABASE_URL:
        conn = _TursoConnection(config.TURSO_DATABASE_URL, config.TURSO_AUTH_TOKEN)
    else:
        conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates the database tables/indexes if they do not already exist."""
    try:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    scan_date TEXT PRIMARY KEY,
                    scan_timestamp TEXT NOT NULL,
                    ai_commentary TEXT,
                    universe_size INTEGER,
                    shortlist_size INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    sector TEXT,
                    close_price REAL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_hist REAL,
                    macd_hist_direction TEXT,
                    nearest_fib_level TEXT,
                    nearest_fib_price REAL,
                    fib_distance_pct REAL,
                    fib_high REAL,
                    fib_low REAL,
                    week52_high REAL,
                    week52_low REAL,
                    pct_from_52w_high REAL,
                    macd_pattern TEXT,
                    volume_ratio REAL,
                    avg_volume_20 REAL,
                    buy_pct REAL,
                    sell_pct REAL,
                    sector_trend_pct REAL,
                    prev_session_date TEXT,
                    prev_session_open REAL,
                    prev_session_close REAL,
                    score INTEGER,
                    reasons TEXT,
                    FOREIGN KEY (scan_date) REFERENCES scans(scan_date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_scan_date
                ON signals(scan_date)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_unique
                ON signals(scan_date, ticker)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_commentary_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    commentary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authorized_users (
                    email TEXT PRIMARY KEY,
                    name TEXT,
                    first_login TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    ai_commentary TEXT,
                    signals_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    user_email TEXT,
                    event TEXT NOT NULL,
                    props TEXT
                )
            """)

            # Migration: add status to a table created by an older version.
            existing_user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(authorized_users)")}
            if "status" not in existing_user_cols:
                conn.execute("ALTER TABLE authorized_users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")

            # Migration: add new columns to a signals table created by an
            # older version of this app, which won't have them yet.
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(signals)")}
            for col, col_type in (
                ("week52_high", "REAL"),
                ("week52_low", "REAL"),
                ("pct_from_52w_high", "REAL"),
                ("macd_pattern", "TEXT"),
                ("volume_ratio", "REAL"),
                ("avg_volume_20", "REAL"),
                ("sector", "TEXT"),
                ("macd_hist_direction", "TEXT"),
                ("buy_pct", "REAL"),
                ("sell_pct", "REAL"),
                ("sector_trend_pct", "REAL"),
                ("prev_session_date", "TEXT"),
                ("prev_session_open", "REAL"),
                ("prev_session_close", "REAL"),
            ):
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to initialize database: {e}")
        raise


def save_scan_results(shortlist_df, ai_commentary, universe_size):
    """
    Persists a scan's shortlist + AI commentary under today's date.

    Re-running the pipeline on the same calendar date deletes the previous
    signals for that date and overwrites the scan row, so the user never
    accumulates duplicate alerts for the same day.

    Returns the scan_date (YYYY-MM-DD) the results were saved under.
    """
    scan_date = datetime.now().strftime("%Y-%m-%d")
    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM signals WHERE scan_date = ?", (scan_date,))
            conn.execute(
                """
                INSERT OR REPLACE INTO scans
                    (scan_date, scan_timestamp, ai_commentary, universe_size, shortlist_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (scan_date, scan_timestamp, ai_commentary, universe_size, len(shortlist_df)),
            )

            for _, row in shortlist_df.iterrows():
                conn.execute(
                    """
                    INSERT INTO signals (
                        scan_date, ticker, sector, close_price, rsi, macd_line, macd_signal,
                        macd_hist, macd_hist_direction, nearest_fib_level, nearest_fib_price,
                        fib_distance_pct, fib_high, fib_low, week52_high, week52_low,
                        pct_from_52w_high, macd_pattern, volume_ratio, avg_volume_20,
                        buy_pct, sell_pct, sector_trend_pct, prev_session_date,
                        prev_session_open, prev_session_close, score, reasons
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_date,
                        row["ticker"],
                        row["sector"],
                        float(row["close"]),
                        float(row["rsi"]),
                        float(row["macd_line"]),
                        float(row["macd_signal"]),
                        float(row["macd_hist"]),
                        row["macd_hist_direction"],
                        row["nearest_fib_level"],
                        float(row["nearest_fib_price"]),
                        float(row["fib_distance_pct"]),
                        float(row["fib_high"]),
                        float(row["fib_low"]),
                        float(row["week52_high"]),
                        float(row["week52_low"]),
                        float(row["pct_from_52w_high"]),
                        row["macd_pattern"],
                        float(row["volume_ratio"]),
                        float(row["avg_volume_20"]),
                        float(row["buy_pct"]),
                        float(row["sell_pct"]),
                        float(row["sector_trend_pct"]),
                        row["prev_session_date"],
                        float(row["prev_session_open"]),
                        float(row["prev_session_close"]),
                        int(row["score"]),
                        json.dumps(row["reasons"]),
                    ),
                )

            # Keep only the SCAN_HISTORY_RETENTION_DAYS most recent scan
            # dates - older scans (and their signals) are deleted so the
            # day-over-day diff and score-history sparkline always have a
            # bounded amount of data to read.
            old_dates = [
                r["scan_date"] for r in conn.execute(
                    "SELECT scan_date FROM scans ORDER BY scan_date DESC"
                ).fetchall()
            ][config.SCAN_HISTORY_RETENTION_DAYS:]
            for old_date in old_dates:
                conn.execute("DELETE FROM signals WHERE scan_date = ?", (old_date,))
                conn.execute("DELETE FROM scans WHERE scan_date = ?", (old_date,))
        return scan_date
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to save scan results: {e}")
        raise


def get_cached_ai_commentary(prompt_hash):
    """
    Returns a previously-generated AI commentary for this exact prompt, but
    only if it was generated earlier TODAY - entries from a previous day are
    treated as stale (a new trading session has started, so even an
    identical-looking prompt should get a fresh take) and are ignored here.
    Returns None on a cache miss (no entry, or entry is from a prior day).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT commentary FROM ai_commentary_cache "
                "WHERE prompt_hash = ? AND created_at LIKE ?",
                (prompt_hash, f"{today}%"),
            ).fetchone()
        return row["commentary"] if row else None
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to read AI commentary cache: {e}")
        return None


def save_ai_commentary_cache(prompt_hash, commentary):
    """
    Stores an AI commentary result so an identical prompt later today can
    reuse it, and deletes any entries left over from previous days (they're
    stale - see get_cached_ai_commentary).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ai_commentary_cache (prompt_hash, commentary, created_at)
                VALUES (?, ?, ?)
                """,
                (prompt_hash, commentary, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.execute(
                "DELETE FROM ai_commentary_cache WHERE created_at NOT LIKE ?",
                (f"{today}%",),
            )
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to write AI commentary cache: {e}")


def _rows_to_signals_df(signal_rows):
    records = []
    for r in signal_rows:
        rec = dict(r)
        try:
            rec["reasons"] = json.loads(rec["reasons"]) if rec["reasons"] else []
        except (TypeError, json.JSONDecodeError):
            rec["reasons"] = []
        # Stored as close_price in SQLite; rename to match strategy.py's "close"
        # so callers can treat live and DB-loaded shortlists identically.
        rec["close"] = rec.pop("close_price")
        records.append(rec)
    return pd.DataFrame(records)


@st.cache_data(ttl=300, show_spinner="Loading latest scan...")
def get_latest_scan():
    """Returns (scan_date, ai_commentary, signals_df) for the most recent scan, or None."""
    try:
        with get_connection() as conn:
            scan_row = conn.execute(
                "SELECT * FROM scans ORDER BY scan_date DESC LIMIT 1"
            ).fetchone()
            if scan_row is None:
                return None

            signal_rows = conn.execute(
                "SELECT * FROM signals WHERE scan_date = ? ORDER BY score DESC",
                (scan_row["scan_date"],),
            ).fetchall()

        return scan_row["scan_date"], scan_row["ai_commentary"], _rows_to_signals_df(signal_rows)
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch latest scan: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_available_scan_dates():
    """Returns all scan dates (YYYY-MM-DD), most recent first."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT scan_date FROM scans ORDER BY scan_date DESC"
            ).fetchall()
        return [r["scan_date"] for r in rows]
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch scan dates: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_scan_timestamps():
    """Returns {scan_date: scan_timestamp} for display labels in the date selector."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT scan_date, scan_timestamp FROM scans ORDER BY scan_date DESC"
            ).fetchall()
        return {r["scan_date"]: r["scan_timestamp"] for r in rows}
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch scan timestamps: {e}")
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_score_history(tickers, as_of_date):
    """
    Returns {ticker: [score, ...]} for each of `tickers`, oldest-first, over
    every retained scan up to and including `as_of_date` (see
    SCAN_HISTORY_RETENTION_DAYS) - used for the Shortlist tab's score-history
    sparkline. Tickers with no history return an empty list.
    """
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    try:
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT scan_date, ticker, score FROM signals "
                f"WHERE ticker IN ({placeholders}) AND scan_date <= ? "
                f"ORDER BY scan_date ASC",
                (*tickers, as_of_date),
            ).fetchall()
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch score history: {e}")
        return {ticker: [] for ticker in tickers}

    history = {ticker: [] for ticker in tickers}
    for r in rows:
        history[r["ticker"]].append(r["score"])
    return history


@st.cache_data(ttl=300, show_spinner="Loading scan...")
def get_scan_by_date(scan_date):
    """Returns (scan_date, ai_commentary, signals_df) for a specific date, or None."""
    try:
        with get_connection() as conn:
            scan_row = conn.execute(
                "SELECT * FROM scans WHERE scan_date = ?", (scan_date,)
            ).fetchone()
            if scan_row is None:
                return None

            signal_rows = conn.execute(
                "SELECT * FROM signals WHERE scan_date = ? ORDER BY score DESC",
                (scan_date,),
            ).fetchall()

        return scan_row["scan_date"], scan_row["ai_commentary"], _rows_to_signals_df(signal_rows)
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch scan for {scan_date}: {e}")
        return None


def is_user_authorized(email):
    """Returns True if this email currently has active access."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM authorized_users WHERE email = ? AND status = 'active'", (email,)
            ).fetchone()
        return row is not None
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to check authorized user {email}: {e}")
        return False


def get_user_status(email):
    """Returns 'active', 'revoked', or None if this email has never registered."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT status FROM authorized_users WHERE email = ?", (email,)
            ).fetchone()
        return row["status"] if row else None
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to get status for {email}: {e}")
        return None


def get_authorized_user_count():
    """Returns the number of emails with currently-active access (toward AUTH_MAX_USERS)."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM authorized_users WHERE status = 'active'"
            ).fetchone()
        return row["n"]
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to count authorized users: {e}")
        return 0


def register_user(email, name):
    """
    Grants a new email active access (counts toward AUTH_MAX_USERS). Safe to
    call even if the email is already registered - existing rows (including
    previously-revoked ones) are left as-is.
    """
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO authorized_users (email, name, first_login, status) "
                "VALUES (?, ?, ?, 'active')",
                (email, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to register user {email}: {e}")
        raise


def get_all_authorized_users():
    """Returns all registered users (active and revoked) as a list of dicts, oldest first."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT email, name, first_login, status FROM authorized_users ORDER BY first_login ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch authorized users: {e}")
        return []


def revoke_user(email):
    """Revokes a user's access - they'll be signed out on their next interaction
    and their slot frees up for someone else."""
    try:
        with get_connection() as conn:
            conn.execute("UPDATE authorized_users SET status = 'revoked' WHERE email = ?", (email,))
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to revoke user {email}: {e}")
        raise


def restore_user(email):
    """Restores a previously-revoked user's access."""
    try:
        with get_connection() as conn:
            conn.execute("UPDATE authorized_users SET status = 'active' WHERE email = ?", (email,))
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to restore user {email}: {e}")
        raise


def save_custom_analysis(tickers, signals_df, ai_commentary):
    """Persists a custom analysis run. Keeps the most recent SCAN_HISTORY_RETENTION_DAYS entries."""
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tickers_json = json.dumps(tickers)
    signals_json = json.dumps(
        signals_df.to_dict(orient="records") if not signals_df.empty else []
    )
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO custom_analyses (run_timestamp, tickers, ai_commentary, signals_json) "
                "VALUES (?, ?, ?, ?)",
                (run_timestamp, tickers_json, ai_commentary, signals_json),
            )
            old_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM custom_analyses ORDER BY id DESC"
                ).fetchall()
            ][config.SCAN_HISTORY_RETENTION_DAYS:]
            for old_id in old_ids:
                conn.execute("DELETE FROM custom_analyses WHERE id = ?", (old_id,))
        return run_timestamp
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to save custom analysis: {e}")
        raise


@st.cache_data(ttl=300, show_spinner=False)
def get_available_custom_analyses():
    """Returns [(id, run_timestamp, tickers_list), ...] most recent first."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, run_timestamp, tickers FROM custom_analyses ORDER BY id DESC"
            ).fetchall()
        return [(r["id"], r["run_timestamp"], json.loads(r["tickers"])) for r in rows]
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch custom analyses: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_custom_analysis_by_id(analysis_id):
    """Returns (tickers, signals_df, ai_commentary) for a saved custom analysis, or None."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM custom_analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        if row is None:
            return None
        tickers = json.loads(row["tickers"])
        records = json.loads(row["signals_json"])
        signals_df = pd.DataFrame(records) if records else pd.DataFrame()
        return tickers, signals_df, row["ai_commentary"]
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to fetch custom analysis {analysis_id}: {e}")
        return None


def log_event(user_email, event, props=None):
    """Fire-and-forget analytics event. Never raises — analytics must never crash the app."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO events (ts, user_email, event, props) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), user_email or "", event, json.dumps(props or {})),
            )
    except Exception as e:
        print(f"[db_handler] log_event failed silently: {e}")


def delete_custom_analysis(analysis_id):
    """Permanently deletes a custom analysis by id."""
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM custom_analyses WHERE id = ?", (analysis_id,))
        get_available_custom_analyses.clear()
        get_custom_analysis_by_id.clear()
    except _DB_ERRORS as e:
        print(f"[db_handler] Failed to delete custom analysis {analysis_id}: {e}")
        raise
