"""
Observability API endpoints for the Unified Dashboard.

Provides REST API access to the ObservabilityDashboard data for
real-time monitoring of agentic search requests.

Updated 2026-01-07: Added database-backed historical endpoints.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from agentic.observability_dashboard import (
    get_observability_dashboard,
    RequestObservability,
    DashboardStats
)
from config.database import get_db_dependency
from models.agent_run import AgentRun, AgentRunStatus

logger = logging.getLogger("api.observability")

router = APIRouter(prefix="/api/v1/observability", tags=["Observability"])


class RecentRequestsResponse(BaseModel):
    """Response model for recent requests endpoint."""
    success: bool
    count: int
    requests: List[dict]


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    success: bool
    stats: dict


class RequestDetailResponse(BaseModel):
    """Response model for individual request details."""
    success: bool
    request: Optional[dict]
    technician_summary: Optional[str]


@router.get("/recent", response_model=RecentRequestsResponse)
async def get_recent_requests(
    limit: int = Query(default=20, ge=1, le=100),
    preset: Optional[str] = Query(default=None),
    success_only: bool = Query(default=False)
):
    """
    Get recent observability data for agent requests.

    Returns the most recent requests with their observability summaries.
    Used by the Unified Dashboard for live monitoring.
    """
    dashboard = get_observability_dashboard()

    recent = dashboard.get_recent_requests(
        limit=limit,
        preset=preset,
        success_only=success_only
    )

    return RecentRequestsResponse(
        success=True,
        count=len(recent),
        requests=[r.to_dict() for r in recent]
    )


@router.get("/stats", response_model=StatsResponse)
async def get_observability_stats(
    last_hours: int = Query(default=1, ge=1, le=24),
    preset: Optional[str] = Query(default=None)
):
    """
    Get aggregate statistics for observability data.

    Returns performance metrics, confidence distributions,
    agent activity, and feature usage over the specified time window.
    """
    dashboard = get_observability_dashboard()

    stats = dashboard.get_stats(
        last_hours=last_hours,
        preset=preset
    )

    return StatsResponse(
        success=True,
        stats=stats.to_dict()
    )


@router.get("/request/{request_id}", response_model=RequestDetailResponse)
async def get_request_observability(request_id: str):
    """
    Get detailed observability data for a specific request.

    Returns the full observability record including decisions,
    LLM calls, context flow, and confidence breakdown.
    """
    dashboard = get_observability_dashboard()

    obs = dashboard.get_request(request_id)

    if not obs:
        return RequestDetailResponse(
            success=False,
            request=None,
            technician_summary=None
        )

    return RequestDetailResponse(
        success=True,
        request=obs.to_dict(),
        technician_summary=obs.to_technician_summary()
    )


@router.get("/request/{request_id}/audit")
async def get_request_audit(request_id: str):
    """
    Get a technician-friendly audit trail for a request.

    Returns a markdown-formatted summary suitable for
    debugging and troubleshooting.
    """
    dashboard = get_observability_dashboard()

    audit = dashboard.get_technician_audit(request_id)

    if not audit:
        return {
            "success": False,
            "audit": None,
            "error": "Request not found"
        }

    return {
        "success": True,
        "audit": audit
    }


@router.get("/health")
async def get_observability_health():
    """
    Get health status of the observability dashboard.

    Returns storage metrics and capacity information.
    """
    dashboard = get_observability_dashboard()
    health = dashboard.get_health()

    return {
        "success": True,
        **health
    }


# ============== Database-Backed Historical Endpoints ==============
# Added 2026-01-07 for persistent Agent Console history


@router.get("/history")
async def get_agent_run_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    preset: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    hours: Optional[int] = Query(default=None, ge=1, le=720),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get historical agent runs from the database.

    This endpoint provides persistent storage for Agent Console,
    allowing users to view past runs even after server restarts.

    Args:
        limit: Maximum number of results (default 50, max 500)
        offset: Pagination offset
        preset: Filter by preset (balanced, enhanced, full, etc.)
        status: Filter by status (running, completed, failed)
        hours: Only return runs from the last N hours
    """
    try:
        query = select(AgentRun).order_by(desc(AgentRun.started_at))

        # Apply filters
        if preset:
            query = query.where(AgentRun.preset == preset)

        if status:
            query = query.where(AgentRun.status == status)

        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = query.where(AgentRun.started_at >= cutoff)

        # Apply pagination
        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        runs = result.scalars().all()

        # Get total count for pagination
        count_query = select(AgentRun)
        if preset:
            count_query = count_query.where(AgentRun.preset == preset)
        if status:
            count_query = count_query.where(AgentRun.status == status)
        if hours:
            count_query = count_query.where(AgentRun.started_at >= cutoff)

        from sqlalchemy import func
        count_result = await db.execute(select(func.count()).select_from(count_query.subquery()))
        total_count = count_result.scalar()

        return {
            "success": True,
            "data": {
                "runs": [run.to_summary() for run in runs],
                "total": total_count,
                "limit": limit,
                "offset": offset,
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "database",
            },
            "errors": []
        }

    except Exception as e:
        logger.error(f"Failed to get agent run history: {e}")
        return {
            "success": False,
            "data": {"runs": [], "total": 0},
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": [str(e)]
        }


@router.get("/history/{request_id}")
async def get_agent_run_detail(
    request_id: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get detailed information for a specific agent run from the database.

    Returns full event timeline, decisions, LLM calls, and all observability
    data for historical analysis.

    First checks in-memory cache, then falls back to database.
    """
    # First check in-memory cache (for very recent requests)
    dashboard = get_observability_dashboard()
    in_memory = dashboard.get_request(request_id)

    if in_memory:
        return {
            "success": True,
            "data": in_memory.to_dict(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "memory",
            },
            "errors": []
        }

    # Fall back to database
    try:
        result = await db.execute(
            select(AgentRun).where(AgentRun.request_id == request_id)
        )
        run = result.scalar_one_or_none()

        if not run:
            return {
                "success": False,
                "data": None,
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "database",
                },
                "errors": ["Request not found"]
            }

        return {
            "success": True,
            "data": run.to_dict(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "database",
            },
            "errors": []
        }

    except Exception as e:
        logger.error(f"Failed to get agent run detail: {e}")
        return {
            "success": False,
            "data": None,
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": [str(e)]
        }


@router.get("/history/{request_id}/events")
async def get_agent_run_events(
    request_id: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get just the events timeline for a specific agent run.

    Returns the events array for displaying in the Agent Console
    Event Stream panel.

    Prefers database events (SSE events with full detail) over in-memory
    events (just LLM calls summary) since database has richer information.
    """
    # Check database first for rich SSE events
    try:
        result = await db.execute(
            select(AgentRun.events, AgentRun.query, AgentRun.preset, AgentRun.status)
            .where(AgentRun.request_id == request_id)
        )
        row = result.one_or_none()

        if row and row[0]:
            db_events = row[0] or []
            if len(db_events) > 0:
                # Database has events, use those (they have full SSE detail)
                return {
                    "success": True,
                    "data": {
                        "request_id": request_id,
                        "events": db_events,
                        "count": len(db_events),
                    },
                    "meta": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "database",
                    },
                    "errors": []
                }
    except Exception as e:
        logger.warning(f"Database events lookup failed: {e}")

    # Fall back to in-memory (for very recent requests not yet persisted)
    dashboard = get_observability_dashboard()
    in_memory = dashboard.get_request(request_id)

    if in_memory:
        events = dashboard._build_events_timeline(in_memory) if hasattr(dashboard, '_build_events_timeline') else []
        return {
            "success": True,
            "data": {
                "request_id": request_id,
                "events": events,
                "count": len(events),
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "memory",
            },
            "errors": []
        }

    # Request not found in either database or memory
    return {
        "success": False,
        "data": {"events": [], "count": 0},
        "errors": ["Request not found"]
    }
