"""
Memory-specific encryption utilities
Wrapper functions for memory service encryption needs
"""

from core.encryption_service import EncryptionService

# Create singleton instance
_encryption_service = EncryptionService()

def encrypt_data(data: str) -> bytes:
    """Encrypt memory content data"""
    return _encryption_service.encrypt_database_field(data, field_name='memory_content').encode()

def decrypt_data(encrypted_data: bytes) -> str:
    """Decrypt memory content data"""
    if isinstance(encrypted_data, bytes):
        encrypted_data = encrypted_data.decode()
    decrypted = _encryption_service.decrypt_database_field(encrypted_data)
    # Extract the value from the JSON context
    import json
    try:
        context = json.loads(decrypted)
        return context.get('value', decrypted)
    except (json.JSONDecodeError, TypeError, KeyError):
        return decrypted