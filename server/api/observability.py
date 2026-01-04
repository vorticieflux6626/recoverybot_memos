"""
Observability API endpoints for the Unified Dashboard.

Provides REST API access to the ObservabilityDashboard data for
real-time monitoring of agentic search requests.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

from agentic.observability_dashboard import (
    get_observability_dashboard,
    RequestObservability,
    DashboardStats
)

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
