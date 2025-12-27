#!/usr/bin/env python3
"""Fix broken memory.py file"""

# Read the broken file
with open('api/memory_broken.py', 'r') as f:
    content = f.read()

# Read the original working file
with open('api/memory.py', 'r') as f:
    broken_content = f.read()

# Backup the broken file
with open('api/memory_broken.py', 'w') as f:
    f.write(broken_content)

# Start fresh with the original content
with open('api/memory.py', 'r') as f:
    content = f.read()

# Apply only the essential fix for audit_logger usage in search_memories
content = content.replace(
    '''        # Audit log the search request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id": user_id,
                "memory_id": "search_operation",
                "operation": "SEARCH",
                "requester_id": user_id,
                "requester_role": "user",
                "ip_address": request.client.host if request and request.client else None,
                "user_agent": request.headers.get("user-agent") if request else None,
                "details": {"query_length": len(query), "include_content": include_content},
                "action": "memory_access"
            }
        )''',
    '''        # Audit log the search request
        audit_logger.log_memory_access(
            user_id=user_id,
            memory_id="search_operation",
            operation="SEARCH",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details={"query_length": len(query), "include_content": include_content}
        )'''
)

# Save the fixed content
with open('api/memory.py', 'w') as f:
    f.write(content)

print("Fixed memory.py - audit logger issue in search_memories")