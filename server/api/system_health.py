"""
System-wide health aggregator.

Reads from SYSTEM_MANIFEST.yaml and checks all subsystems.
Provides a single endpoint for system-wide health monitoring.

Endpoints:
    GET /api/v1/system/health/aggregate - Check health of all subsystems
    GET /api/v1/system/manifest - Get current system manifest
    GET /api/v1/system/topology - Get system topology graph
"""

import yaml
import httpx
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("api.system_health")

router = APIRouter(prefix="/api/v1/system", tags=["system"])

# Path to SYSTEM_MANIFEST.yaml (3 levels up from this file to Recovery_Bot/)
MANIFEST_PATH = Path(__file__).parents[3] / ".memOS" / "SYSTEM_MANIFEST.yaml"


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    NO_HEALTH_CHECK = "no_health_check"


@dataclass
class SubsystemHealth:
    """Health status of a single subsystem."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    port: Optional[int] = None
    health_endpoint: Optional[str] = None
    last_checked: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "port": self.port,
            "health_endpoint": self.health_endpoint,
            "last_checked": self.last_checked
        }


async def check_subsystem_health(
    name: str,
    config: Dict[str, Any],
    timeout: float = 5.0
) -> SubsystemHealth:
    """
    Check health of a single subsystem.

    Args:
        name: Subsystem name
        config: Subsystem configuration from manifest
        timeout: Request timeout in seconds

    Returns:
        SubsystemHealth object with status and metrics
    """
    health_endpoint = config.get("health_endpoint")
    port = config.get("port")

    # If no health endpoint configured, return unknown status
    if not health_endpoint or not port:
        return SubsystemHealth(
            name=name,
            status=HealthStatus.NO_HEALTH_CHECK,
            port=port,
            health_endpoint=health_endpoint,
            last_checked=datetime.now(timezone.utc).isoformat()
        )

    url = f"http://localhost:{port}{health_endpoint}"

    try:
        start = datetime.now(timezone.utc)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        status = HealthStatus.HEALTHY if response.status_code == 200 else HealthStatus.DEGRADED

        return SubsystemHealth(
            name=name,
            status=status,
            latency_ms=round(latency, 2),
            port=port,
            health_endpoint=health_endpoint,
            last_checked=datetime.now(timezone.utc).isoformat()
        )

    except httpx.TimeoutException:
        return SubsystemHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            error="Connection timeout",
            port=port,
            health_endpoint=health_endpoint,
            last_checked=datetime.now(timezone.utc).isoformat()
        )

    except httpx.ConnectError as e:
        return SubsystemHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            error=f"Connection refused",
            port=port,
            health_endpoint=health_endpoint,
            last_checked=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        return SubsystemHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            error=str(e),
            port=port,
            health_endpoint=health_endpoint,
            last_checked=datetime.now(timezone.utc).isoformat()
        )


def load_manifest() -> Optional[Dict[str, Any]]:
    """Load SYSTEM_MANIFEST.yaml."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        return yaml.safe_load(MANIFEST_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        return None


def determine_overall_status(results: List[SubsystemHealth]) -> HealthStatus:
    """Determine overall system health from subsystem results."""
    statuses = [r.status for r in results if r.status != HealthStatus.NO_HEALTH_CHECK]

    if not statuses:
        return HealthStatus.UNKNOWN

    if all(s == HealthStatus.HEALTHY for s in statuses):
        return HealthStatus.HEALTHY

    if any(s == HealthStatus.UNHEALTHY for s in statuses):
        return HealthStatus.UNHEALTHY

    return HealthStatus.DEGRADED


@router.get("/health/aggregate")
async def aggregate_health():
    """
    Check health of all subsystems defined in SYSTEM_MANIFEST.yaml.

    Returns:
        Unified view of system health with per-subsystem status.
    """
    manifest = load_manifest()

    if not manifest:
        return {
            "success": False,
            "data": None,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "manifest_path": str(MANIFEST_PATH),
                "manifest_exists": MANIFEST_PATH.exists()
            },
            "errors": [{
                "code": "SYS_001",
                "message": "SYSTEM_MANIFEST.yaml not found or invalid"
            }]
        }

    subsystems = manifest.get("subsystems", {})
    external_services = manifest.get("external_services", {})

    # Check all subsystems and external services in parallel
    tasks = []

    for name, config in subsystems.items():
        tasks.append(check_subsystem_health(name, config))

    for name, config in external_services.items():
        tasks.append(check_subsystem_health(name, config))

    results = await asyncio.gather(*tasks)

    # Determine overall status
    overall_status = determine_overall_status(results)

    # Separate results by type
    subsystem_results = [r.to_dict() for r in results[:len(subsystems)]]
    external_results = [r.to_dict() for r in results[len(subsystems):]]

    return {
        "success": True,
        "data": {
            "overall_status": overall_status.value,
            "subsystems": subsystem_results,
            "external_services": external_results,
            "summary": {
                "total_checked": len(results),
                "healthy": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
                "no_health_check": sum(1 for r in results if r.status == HealthStatus.NO_HEALTH_CHECK)
            },
            "manifest_version": manifest.get("system", {}).get("version")
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(MANIFEST_PATH)
        },
        "errors": []
    }


@router.get("/manifest")
async def get_manifest():
    """
    Get the current SYSTEM_MANIFEST.yaml contents.

    Returns:
        The full manifest as JSON.
    """
    manifest = load_manifest()

    if not manifest:
        return {
            "success": False,
            "data": None,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "manifest_path": str(MANIFEST_PATH)
            },
            "errors": [{
                "code": "SYS_001",
                "message": "SYSTEM_MANIFEST.yaml not found or invalid"
            }]
        }

    return {
        "success": True,
        "data": manifest,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(MANIFEST_PATH)
        },
        "errors": []
    }


@router.get("/topology")
async def get_topology():
    """
    Get system topology as a graph structure.

    Returns:
        Nodes (subsystems) and edges (dependencies) for visualization.
    """
    manifest = load_manifest()

    if not manifest:
        return {
            "success": False,
            "data": None,
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": [{"code": "SYS_001", "message": "Manifest not found"}]
        }

    nodes = []
    edges = []

    # Build nodes from subsystems
    for name, config in manifest.get("subsystems", {}).items():
        nodes.append({
            "id": name,
            "type": "subsystem",
            "subtype": config.get("type"),
            "port": config.get("port"),
            "repo": config.get("repo"),
            "ssot_for": config.get("ssot_for", [])
        })

        # Build edges from dependencies
        for dep in config.get("dependencies", []):
            edges.append({
                "from": name,
                "to": dep,
                "type": "depends_on"
            })

    # Add external services as nodes
    for name, config in manifest.get("external_services", {}).items():
        nodes.append({
            "id": name,
            "type": "external_service",
            "port": config.get("port"),
            "required": config.get("required", False)
        })

    # Add contract edges
    for contract in manifest.get("contracts", []):
        provider = contract.get("provider")
        for consumer in contract.get("consumers", []):
            edges.append({
                "from": consumer,
                "to": provider,
                "type": "consumes_api",
                "contract": contract.get("path")
            })

    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": manifest.get("system", {}).get("version")
        },
        "errors": []
    }


@router.get("/health/subsystem/{subsystem_name}")
async def check_single_subsystem(subsystem_name: str):
    """
    Check health of a specific subsystem.

    Args:
        subsystem_name: Name of the subsystem to check

    Returns:
        Health status of the specified subsystem.
    """
    manifest = load_manifest()

    if not manifest:
        raise HTTPException(status_code=500, detail="Manifest not found")

    subsystems = manifest.get("subsystems", {})
    external = manifest.get("external_services", {})

    config = subsystems.get(subsystem_name) or external.get(subsystem_name)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Subsystem '{subsystem_name}' not found in manifest"
        )

    result = await check_subsystem_health(subsystem_name, config)

    return {
        "success": True,
        "data": result.to_dict(),
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "errors": []
    }
