"""
Agentic Search Schemas

Domain-specific entity schemas for technical documentation processing.

Domains:
- FANUC Robotics: Error codes, components, procedures, parameters
- IMM (Injection Molding Machines): KraussMaffei, Milacron, Van Dorn
- Euromap Protocols: 67, 73, 77 robot-IMM interface standards
- RJG Scientific Molding: Cavity pressure, decoupled molding, defects
- PLC/Automation: Allen-Bradley, Siemens, AutomationDirect
"""

from .fanuc_schema import (
    FANUC_SCHEMA,
    FANUC_ERROR_PATTERNS,
    FANUC_COMPONENT_PATTERNS,
    FANUC_PROCEDURE_PATTERNS,
    FANUC_PARAMETER_PATTERNS,
    create_fanuc_domain_schema
)

from .rjg_schema import (
    RJG_SCHEMA,
    RJGEntityType,
    RJGRelationType,
    RJG_PRODUCT_PATTERNS,
    RJG_PROCESS_PHASE_PATTERNS,
    RJG_PROCESS_VARIABLE_PATTERNS,
    RJG_DEFECT_PATTERNS,
    RJG_SENSOR_PATTERNS,
    RJG_TECHNIQUE_PATTERNS,
    RJG_ACRONYMS,
    create_rjg_domain_schema,
    is_scientific_molding_query,
    extract_defects,
    get_defect_category,
    extract_process_variables,
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

from .plc_schema import (
    PLC_SCHEMA,
    PLCEntityType,
    PLCRelationType,
    PLC_ACRONYMS,
    AB_FAULT_PATTERNS,
    AB_ADDRESS_PATTERNS,
    AB_MODULE_PATTERNS,
    SIEMENS_FAULT_PATTERNS,
    SIEMENS_ADDRESS_PATTERNS,
    SIEMENS_MODULE_PATTERNS,
    AD_FAULT_PATTERNS,
    AD_ADDRESS_PATTERNS,
    AD_MODULE_PATTERNS,
    PROTOCOL_PATTERNS,
    HMI_PATTERNS,
    INSTRUCTION_PATTERNS,
    is_plc_query,
    detect_plc_manufacturer,
    extract_fault_codes,
    extract_module_numbers,
    get_fault_description,
    create_plc_domain_schema,
)

__all__ = [
    # FANUC
    "FANUC_SCHEMA",
    "FANUC_ERROR_PATTERNS",
    "FANUC_COMPONENT_PATTERNS",
    "FANUC_PROCEDURE_PATTERNS",
    "FANUC_PARAMETER_PATTERNS",
    "create_fanuc_domain_schema",
    # RJG Scientific Molding
    "RJG_SCHEMA",
    "RJGEntityType",
    "RJGRelationType",
    "RJG_PRODUCT_PATTERNS",
    "RJG_PROCESS_PHASE_PATTERNS",
    "RJG_PROCESS_VARIABLE_PATTERNS",
    "RJG_DEFECT_PATTERNS",
    "RJG_SENSOR_PATTERNS",
    "RJG_TECHNIQUE_PATTERNS",
    "RJG_ACRONYMS",
    "create_rjg_domain_schema",
    "is_scientific_molding_query",
    "extract_defects",
    "get_defect_category",
    "extract_process_variables",
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
    # PLC/Automation
    "PLC_SCHEMA",
    "PLCEntityType",
    "PLCRelationType",
    "PLC_ACRONYMS",
    "AB_FAULT_PATTERNS",
    "AB_ADDRESS_PATTERNS",
    "AB_MODULE_PATTERNS",
    "SIEMENS_FAULT_PATTERNS",
    "SIEMENS_ADDRESS_PATTERNS",
    "SIEMENS_MODULE_PATTERNS",
    "AD_FAULT_PATTERNS",
    "AD_ADDRESS_PATTERNS",
    "AD_MODULE_PATTERNS",
    "PROTOCOL_PATTERNS",
    "HMI_PATTERNS",
    "INSTRUCTION_PATTERNS",
    "is_plc_query",
    "detect_plc_manufacturer",
    "extract_fault_codes",
    "extract_module_numbers",
    "get_fault_description",
    "create_plc_domain_schema",
]
