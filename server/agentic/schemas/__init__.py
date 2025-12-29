"""
Agentic Search Schemas

Domain-specific entity schemas for technical documentation processing.

Domains:
- FANUC Robotics: Error codes, components, procedures, parameters
- IMM (Injection Molding Machines): KraussMaffei, Milacron, Van Dorn
- Euromap Protocols: 67, 73, 77 robot-IMM interface standards
"""

from .fanuc_schema import (
    FANUC_SCHEMA,
    FANUC_ERROR_PATTERNS,
    FANUC_COMPONENT_PATTERNS,
    FANUC_PROCEDURE_PATTERNS,
    FANUC_PARAMETER_PATTERNS,
    create_fanuc_domain_schema
)

from .imm_schema import (
    IMM_SCHEMA,
    IMMEntityType,
    IMMRelationType,
    EUROMAP_PROTOCOL_PATTERNS,
    EUROMAP_SIGNAL_PATTERNS,
    KRAUSSMAFFEI_MODEL_PATTERNS,
    MILACRON_MODEL_PATTERNS,
    VANDORN_MODEL_PATTERNS,
    DEFECT_PATTERNS,
    PROCESS_VARIABLE_PATTERNS,
    RJG_PATTERNS,
    IMM_URL_SOURCES,
    create_imm_domain_schema,
    is_imm_query,
    detect_manufacturer,
    extract_defect_types,
    extract_euromap_signals,
    get_priority_urls,
)

__all__ = [
    # FANUC
    "FANUC_SCHEMA",
    "FANUC_ERROR_PATTERNS",
    "FANUC_COMPONENT_PATTERNS",
    "FANUC_PROCEDURE_PATTERNS",
    "FANUC_PARAMETER_PATTERNS",
    "create_fanuc_domain_schema",
    # IMM
    "IMM_SCHEMA",
    "IMMEntityType",
    "IMMRelationType",
    "EUROMAP_PROTOCOL_PATTERNS",
    "EUROMAP_SIGNAL_PATTERNS",
    "KRAUSSMAFFEI_MODEL_PATTERNS",
    "MILACRON_MODEL_PATTERNS",
    "VANDORN_MODEL_PATTERNS",
    "DEFECT_PATTERNS",
    "PROCESS_VARIABLE_PATTERNS",
    "RJG_PATTERNS",
    "IMM_URL_SOURCES",
    "create_imm_domain_schema",
    "is_imm_query",
    "detect_manufacturer",
    "extract_defect_types",
    "extract_euromap_signals",
    "get_priority_urls",
]
