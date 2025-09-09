# db.py
import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = "memory/history.db"

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_histories(query: str = "") -> List[Dict[str, Any]]:
    """
    히스토리 목록 검색. query가 있으면 q 또는 city에 LIKE 검색.
    최신 생성 순으로 정렬.
    """
    con = _conn()
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if query.strip():
        like = f"%{query.strip()}%"
        cur.execute(
            """
            SELECT rowid AS id, q, a, city, days, nights, ppl, with_kids, ts
            FROM history
            WHERE (q LIKE ? OR city LIKE ?)
            ORDER BY ts DESC
            """,
            (like, like),
        )
    else:
        cur.execute(
            """
            SELECT rowid AS id, q, a, city, days, nights, ppl, with_kids, ts
            FROM history
            ORDER BY ts DESC
            """
        )

    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def get_history_by_id(hid: int) -> Optional[Dict[str, Any]]:
    con = _conn()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT rowid AS id, q, a, city, days, nights, ppl, with_kids, ts, profile_json, weather_json, constraints_json FROM history WHERE rowid=?",
        (hid,),
    )
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None

def delete_history_by_id(hid: int) -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute("DELETE FROM history WHERE rowid=?", (hid,))
    con.commit()
    con.close()

def delete_all_histories() -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute("DELETE FROM history")
    con.commit()
    con.close()
