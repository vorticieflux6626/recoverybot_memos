#!/usr/bin/env python3
"""
Test Results Database - Persistent storage for agentic pipeline test findings.

Usage:
    from tests.data.test_results import TestResultsDB

    db = TestResultsDB()

    # Store a test result
    db.add_result(
        query="FANUC SRVO-023 alarm...",
        preset="enhanced",
        duration_s=45.2,
        confidence=0.58,
        source_count=10,
        synthesis_preview="The servo motor overload...",
        issues=["off_topic_sources", "vram_exhaustion"],
        notes="VL scraper failed due to VRAM"
    )

    # Query results
    results = db.get_results(preset="enhanced", min_confidence=0.5)

    # Get issues summary
    issues = db.get_issue_frequency()
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class TestResultsDB:
    """SQLite database for persistent test result storage."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "test_results.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    query_domain TEXT,
                    preset TEXT NOT NULL,
                    duration_s REAL,
                    confidence REAL,
                    source_count INTEGER,
                    relevant_source_count INTEGER,
                    synthesis_preview TEXT,
                    full_response TEXT,
                    sources_json TEXT,
                    issues_json TEXT,
                    notes TEXT,
                    vram_used_mb INTEGER,
                    gpu_util_pct INTEGER,
                    cache_hit INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS test_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_result_id INTEGER,
                    log_level TEXT,
                    component TEXT,
                    message TEXT,
                    FOREIGN KEY (test_result_id) REFERENCES test_results(id)
                );

                CREATE TABLE IF NOT EXISTS issue_tracker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_code TEXT UNIQUE NOT NULL,
                    description TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'open',
                    resolution TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_results_preset ON test_results(preset);
                CREATE INDEX IF NOT EXISTS idx_results_timestamp ON test_results(timestamp);
                CREATE INDEX IF NOT EXISTS idx_results_domain ON test_results(query_domain);
                CREATE INDEX IF NOT EXISTS idx_logs_component ON test_logs(component);
                CREATE INDEX IF NOT EXISTS idx_issues_code ON issue_tracker(issue_code);
            """)

    def add_result(
        self,
        query: str,
        preset: str,
        duration_s: Optional[float] = None,
        confidence: Optional[float] = None,
        source_count: Optional[int] = None,
        relevant_source_count: Optional[int] = None,
        synthesis_preview: Optional[str] = None,
        full_response: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        issues: Optional[List[str]] = None,
        notes: Optional[str] = None,
        query_domain: Optional[str] = None,
        vram_used_mb: Optional[int] = None,
        gpu_util_pct: Optional[int] = None,
        cache_hit: bool = False,
        success: bool = True
    ) -> int:
        """Add a test result to the database."""
        timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO test_results (
                    timestamp, query, query_domain, preset, duration_s, confidence,
                    source_count, relevant_source_count, synthesis_preview, full_response,
                    sources_json, issues_json, notes, vram_used_mb, gpu_util_pct,
                    cache_hit, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, query, query_domain, preset, duration_s, confidence,
                source_count, relevant_source_count, synthesis_preview, full_response,
                json.dumps(sources) if sources else None,
                json.dumps(issues) if issues else None,
                notes, vram_used_mb, gpu_util_pct,
                1 if cache_hit else 0, 1 if success else 0
            ))
            result_id = cursor.lastrowid

            # Track issues
            if issues:
                for issue_code in issues:
                    self._track_issue(conn, issue_code, timestamp)

            return result_id

    def _track_issue(self, conn, issue_code: str, timestamp: str):
        """Track issue occurrence."""
        conn.execute("""
            INSERT INTO issue_tracker (issue_code, first_seen, last_seen, occurrence_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(issue_code) DO UPDATE SET
                last_seen = excluded.last_seen,
                occurrence_count = occurrence_count + 1
        """, (issue_code, timestamp, timestamp))

    def add_log(
        self,
        message: str,
        test_result_id: Optional[int] = None,
        log_level: str = "INFO",
        component: Optional[str] = None
    ):
        """Add a log entry."""
        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO test_logs (timestamp, test_result_id, log_level, component, message)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, test_result_id, log_level, component, message))

    def get_results(
        self,
        preset: Optional[str] = None,
        query_domain: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        success_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query test results with filters."""
        conditions = []
        params = []

        if preset:
            conditions.append("preset = ?")
            params.append(preset)
        if query_domain:
            conditions.append("query_domain = ?")
            params.append(query_domain)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)
        if success_only:
            conditions.append("success = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM test_results
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params)

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('sources_json'):
                    result['sources'] = json.loads(result['sources_json'])
                if result.get('issues_json'):
                    result['issues'] = json.loads(result['issues_json'])
                results.append(result)
            return results

    def get_issue_frequency(self) -> List[Dict[str, Any]]:
        """Get issue frequency summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT issue_code, description, occurrence_count, first_seen, last_seen, status
                FROM issue_tracker
                ORDER BY occurrence_count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_preset_stats(self) -> List[Dict[str, Any]]:
        """Get aggregated stats by preset."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    preset,
                    COUNT(*) as test_count,
                    AVG(duration_s) as avg_duration,
                    AVG(confidence) as avg_confidence,
                    AVG(source_count) as avg_sources,
                    SUM(cache_hit) as cache_hits,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
                FROM test_results
                GROUP BY preset
                ORDER BY test_count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def update_issue(self, issue_code: str, status: str = None, resolution: str = None, description: str = None):
        """Update issue status or resolution."""
        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)
        if resolution:
            updates.append("resolution = ?")
            params.append(resolution)
        if description:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return

        params.append(issue_code)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                UPDATE issue_tracker SET {', '.join(updates)}
                WHERE issue_code = ?
            """, params)

    def export_to_json(self, output_path: str):
        """Export all data to JSON for backup."""
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "results": self.get_results(limit=10000),
            "issues": self.get_issue_frequency(),
            "preset_stats": self.get_preset_stats()
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return output_path


# Convenience functions for quick access
_db = None

def get_db() -> TestResultsDB:
    """Get singleton database instance."""
    global _db
    if _db is None:
        _db = TestResultsDB()
    return _db

def log_test_result(**kwargs) -> int:
    """Quick function to log a test result."""
    return get_db().add_result(**kwargs)

def log_message(message: str, **kwargs):
    """Quick function to log a message."""
    get_db().add_log(message, **kwargs)


if __name__ == "__main__":
    # Demo usage
    db = TestResultsDB()

    # Add sample result
    result_id = db.add_result(
        query="FANUC R-30iB SRVO-023 alarm during arc welding",
        preset="enhanced",
        query_domain="fanuc_robotics",
        duration_s=45.2,
        confidence=0.58,
        source_count=10,
        relevant_source_count=6,
        synthesis_preview="The SRVO-023 alarm indicates servo motor overload...",
        issues=["off_topic_sources", "vram_exhaustion", "vl_scraper_timeout"],
        notes="VL scraper failed due to VRAM - deepseek-r1:14b using 20GB",
        vram_used_mb=21524,
        gpu_util_pct=95
    )
    print(f"Added result ID: {result_id}")

    # Get stats
    print("\nPreset Stats:")
    for stat in db.get_preset_stats():
        print(f"  {stat['preset']}: {stat['test_count']} tests, avg confidence: {stat['avg_confidence']:.2f}")

    print("\nIssue Frequency:")
    for issue in db.get_issue_frequency():
        print(f"  {issue['issue_code']}: {issue['occurrence_count']} occurrences")
