#!/usr/bin/env python3
"""
Fix audit_logger.info calls in memory.py
"""

import re

# Read the file
with open('api/memory.py', 'r') as f:
    content = f.read()

# Pattern to find audit_logger.info calls that need fixing
pattern = r'audit_logger\.info\(\s*f"Memory access",\s*extra=\{\s*user_id'

# Replace with proper format
def fix_audit_logger(match):
    # Get the full match to preserve indentation
    return match.group(0).replace('extra={\n            user_id', 'extra={\n                "user_id"')

# Apply fixes
content = re.sub(pattern, fix_audit_logger, content)

# Fix missing quotes around keys
content = re.sub(r'(\s+)(user_id|memory_id|operation|requester_id|requester_role|ip_address|user_agent|details):', r'\1"\2":', content)

# Add missing action key before closing braces in audit_logger.info calls
# Find patterns like }\n        ) that follow audit_logger.info
pattern = r'(audit_logger\.info\([^)]+extra=\{[^}]+)(\}\s*\))'
replacement = r'\1,\n                "action": "memory_access"\n            \2'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('api/memory.py', 'w') as f:
    f.write(content)

print("Fixed audit_logger.info calls in api/memory.py")