#!/usr/bin/env python3
"""
Log Capture Utility - Captures and stores server logs for analysis.

Usage:
    # Capture recent logs
    python capture_logs.py --recent 100

    # Capture logs with specific patterns
    python capture_logs.py --grep "ERROR|WARNING" --recent 500

    # Capture full session and store
    python capture_logs.py --session "preset_test_2026-01-03"
"""

import argparse
import re
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Server log locations
SERVER_LOGS = [
    "/tmp/memos_server.log",
    "/home/sparkone/sdd/Recovery_Bot/memOS/server/logs/memos_server.log",
    "/home/sparkone/sdd/Recovery_Bot/memOS/server/logs/errors.log"
]


def capture_recent_logs(lines: int = 100, grep_pattern: Optional[str] = None) -> List[str]:
    """Capture recent lines from server logs."""
    captured = []

    for log_path in SERVER_LOGS:
        path = Path(log_path)
        if not path.exists():
            continue

        try:
            with open(path, 'r') as f:
                all_lines = f.readlines()
                recent = all_lines[-lines:] if len(all_lines) > lines else all_lines

                if grep_pattern:
                    pattern = re.compile(grep_pattern, re.IGNORECASE)
                    recent = [l for l in recent if pattern.search(l)]

                captured.extend(recent)
        except Exception as e:
            captured.append(f"[ERROR reading {log_path}]: {e}\n")

    return captured


def parse_log_line(line: str) -> Optional[Dict]:
    """Parse a log line into structured data."""
    # Pattern: 2026-01-03 17:24:00 [INFO] component: message
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] ([^:]+): (.+)'
    match = re.match(pattern, line.strip())

    if match:
        return {
            "timestamp": match.group(1),
            "level": match.group(2),
            "component": match.group(3),
            "message": match.group(4)
        }
    return None


def save_session_log(session_name: str, logs: List[str], metadata: Optional[Dict] = None) -> Path:
    """Save a session log with metadata."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_name}_{timestamp}.json"
    output_path = LOG_DIR / filename

    # Parse logs
    parsed_logs = []
    for line in logs:
        parsed = parse_log_line(line)
        if parsed:
            parsed_logs.append(parsed)
        else:
            parsed_logs.append({"raw": line.strip()})

    data = {
        "session_name": session_name,
        "captured_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
        "log_count": len(parsed_logs),
        "logs": parsed_logs
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def extract_pipeline_events(logs: List[str]) -> List[Dict]:
    """Extract pipeline stage events from logs."""
    events = []
    request_pattern = r'\[([a-f0-9-]+)\]'

    for line in logs:
        # Look for pipeline graph updates
        if "Graph:" in line or "[A" in line or "[Σ" in line:
            match = re.search(request_pattern, line)
            request_id = match.group(1) if match else None
            events.append({
                "type": "pipeline_stage",
                "request_id": request_id,
                "line": line.strip()
            })

        # Look for errors
        elif "[ERROR]" in line or "[WARNING]" in line:
            events.append({
                "type": "error" if "[ERROR]" in line else "warning",
                "line": line.strip()
            })

        # Look for scrape results
        elif "[SCRAPE]" in line:
            events.append({
                "type": "scrape",
                "line": line.strip()
            })

    return events


def get_gpu_snapshot() -> Dict:
    """Capture current GPU state."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "memory_used_mb": int(parts[0]),
                "memory_total_mb": int(parts[1]),
                "utilization_pct": int(parts[2]),
                "temperature_c": int(parts[3])
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


def get_ollama_models() -> List[Dict]:
    """Get currently loaded Ollama models."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/ps"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("models", [])
    except Exception:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Capture and store server logs")
    parser.add_argument("--recent", type=int, default=100, help="Number of recent lines to capture")
    parser.add_argument("--grep", type=str, help="Regex pattern to filter logs")
    parser.add_argument("--session", type=str, help="Session name for saving")
    parser.add_argument("--with-gpu", action="store_true", help="Include GPU snapshot")
    parser.add_argument("--print", action="store_true", help="Print logs to stdout")
    args = parser.parse_args()

    # Capture logs
    logs = capture_recent_logs(args.recent, args.grep)

    if args.print:
        for line in logs:
            print(line.rstrip())
        return

    # Build metadata
    metadata = {
        "lines_requested": args.recent,
        "grep_pattern": args.grep,
        "capture_time": datetime.utcnow().isoformat()
    }

    if args.with_gpu:
        metadata["gpu_snapshot"] = get_gpu_snapshot()
        metadata["ollama_models"] = get_ollama_models()

    # Save if session name provided
    if args.session:
        output_path = save_session_log(args.session, logs, metadata)
        print(f"Saved session log to: {output_path}")

        # Also extract events
        events = extract_pipeline_events(logs)
        events_path = LOG_DIR / f"{args.session}_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(events_path, 'w') as f:
            json.dump(events, f, indent=2)
        print(f"Saved events to: {events_path}")
    else:
        # Just print summary
        print(f"Captured {len(logs)} log lines")
        events = extract_pipeline_events(logs)
        errors = [e for e in events if e["type"] == "error"]
        warnings = [e for e in events if e["type"] == "warning"]
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")

        if metadata.get("gpu_snapshot"):
            gpu = metadata["gpu_snapshot"]
            print(f"  GPU: {gpu.get('memory_used_mb', 'N/A')}MB / {gpu.get('memory_total_mb', 'N/A')}MB")


if __name__ == "__main__":
    main()
