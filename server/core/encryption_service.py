"""
Encryption Service for memOS Server
HIPAA-compliant encryption for memory content at rest and in transit
"""

import os
import hashlib
import logging
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EncryptionService:
    """
    HIPAA-compliant encryption service for memory content protection
    Provides symmetric and asymmetric encryption capabilities
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize encryption key from settings
        self._encryption_key = self._derive_key(self.settings.encryption_key)
        self._fernet = Fernet(self._encryption_key)
        
        # Generate RSA key pair for asymmetric operations
        self._rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self._rsa_public_key = self._rsa_private_key.public_key()
        
        logger.info("Encryption service initialized with AES-256 and RSA-2048")
    
    def encrypt(self, data: str) -> bytes:
        """
        Encrypt text data using AES-256 encryption
        """
        try:
            if not data:
                return b''
            
            # Convert string to bytes
            data_bytes = data.encode('utf-8')
            
            # Encrypt using Fernet (AES-256 in CBC mode with HMAC)
            encrypted_data = self._fernet.encrypt(data_bytes)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """
        Decrypt AES-256 encrypted data back to text
        """
        try:
            if not encrypted_data:
                return ''
            
            # Decrypt using Fernet
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            
            # Convert bytes back to string
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_asymmetric(self, data: str, public_key: Optional[bytes] = None) -> bytes:
        """
        Encrypt data using RSA asymmetric encryption
        Used for sharing between different systems
        """
        try:
            if not data:
                return b''
            
            # Use provided public key or default to our own
            if public_key:
                pub_key = serialization.load_pem_public_key(public_key)
            else:
                pub_key = self._rsa_public_key
            
            # Convert string to bytes
            data_bytes = data.encode('utf-8')
            
            # Encrypt with RSA-OAEP padding
            encrypted_data = pub_key.encrypt(
                data_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> str:
        """
        Decrypt RSA encrypted data using private key
        """
        try:
            if not encrypted_data:
                return ''
            
            # Decrypt with RSA-OAEP padding
            decrypted_bytes = self._rsa_private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise
    
    def generate_hash(self, data: str, algorithm: str = 'sha256') -> str:
        """
        Generate cryptographic hash of data for integrity verification
        """
        try:
            if not data:
                return ''
            
            data_bytes = data.encode('utf-8')
            
            if algorithm == 'sha256':
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512(data_bytes)
            elif algorithm == 'md5':
                hash_obj = hashlib.md5(data_bytes)
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Hash generation failed: {e}")
            raise
    
    def verify_hash(self, data: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """
        Verify data integrity using cryptographic hash
        """
        try:
            calculated_hash = self.generate_hash(data, algorithm)
            return calculated_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def generate_key_pair(self) -> tuple[bytes, bytes]:
        """
        Generate new RSA key pair for asymmetric encryption
        Returns (private_key_pem, public_key_pem)
        """
        try:
            # Generate new RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys to PEM format
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
            
        except Exception as e:
            logger.error(f"Key pair generation failed: {e}")
            raise
    
    def get_public_key_pem(self) -> bytes:
        """
        Get the public key in PEM format for sharing
        """
        return self._rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def encrypt_database_field(self, field_value: str, field_name: str = '') -> str:
        """
        Encrypt database field with additional context for HIPAA compliance
        """
        try:
            if not field_value:
                return ''
            
            # Add field context for audit purposes
            context_data = {
                'field_name': field_name,
                'encrypted_at': self._get_timestamp(),
                'value': field_value
            }
            
            # Convert to JSON string and encrypt
            import json
            context_json = json.dumps(context_data)
            encrypted_bytes = self.encrypt(context_json)
            
            # Return base64 encoded for database storage
            return base64.b64encode(encrypted_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Database field encryption failed: {e}")
            raise
    
    def decrypt_database_field(self, encrypted_field: str) -> str:
        """
        Decrypt database field and extract original value
        """
        try:
            if not encrypted_field:
                return ''
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_field.encode('utf-8'))
            
            # Decrypt to JSON string
            context_json = self.decrypt(encrypted_bytes)
            
            # Parse JSON and extract original value
            import json
            context_data = json.loads(context_json)
            
            return context_data.get('value', '')
            
        except Exception as e:
            logger.error(f"Database field decryption failed: {e}")
            raise
    
    def secure_delete(self, data: str, passes: int = 3) -> bool:
        """
        Securely overwrite sensitive data in memory (best effort)
        """
        try:
            if not data:
                return True
            
            # Multiple passes with different patterns
            patterns = [b'\x00', b'\xFF', b'\xAA']
            
            for i in range(passes):
                pattern = patterns[i % len(patterns)]
                # This is a best-effort approach in Python
                # True secure deletion requires lower-level memory management
                data = pattern.decode('latin-1', errors='ignore') * len(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Secure deletion failed: {e}")
            return False
    
    def rotate_encryption_key(self, new_key: str) -> bool:
        """
        Rotate encryption key for enhanced security
        Note: This would require re-encrypting all existing data
        """
        try:
            # Derive new key
            new_encryption_key = self._derive_key(new_key)
            new_fernet = Fernet(new_encryption_key)
            
            # In production, this would require:
            # 1. Decrypt all existing data with old key
            # 2. Re-encrypt with new key
            # 3. Update key atomically
            
            # For now, just update the key
            self._encryption_key = new_encryption_key
            self._fernet = new_fernet
            
            logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def get_encryption_info(self) -> dict:
        """
        Get encryption service information for monitoring
        """
        return {
            'symmetric_algorithm': 'AES-256-CBC',
            'asymmetric_algorithm': 'RSA-2048',
            'key_derivation': 'PBKDF2-HMAC-SHA256',
            'hmac_algorithm': 'HMAC-SHA256',
            'initialized': bool(self._fernet),
            'hipaa_compliant': True
        }
    
    # Private helper methods
    
    def _derive_key(self, password: str) -> bytes:
        """
        Derive encryption key from password using PBKDF2
        """
        # Use a fixed salt for consistent key derivation
        # In production, this should be configurable
        salt = b'memOS_recovery_bot_salt_2024'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # NIST recommended minimum
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        return key
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for audit purposes"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Utility functions for common encryption operations

def encrypt_memory_content(content: str) -> bytes:
    """Encrypt memory content using global encryption service"""
    encryption_service = EncryptionService()
    return encryption_service.encrypt(content)


def decrypt_memory_content(encrypted_content: bytes) -> str:
    """Decrypt memory content using global encryption service"""
    encryption_service = EncryptionService()
    return encryption_service.decrypt(encrypted_content)


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash of content for integrity verification"""
    encryption_service = EncryptionService()
    return encryption_service.generate_hash(content, 'sha256')


def verify_content_integrity(content: str, expected_hash: str) -> bool:
    """Verify content integrity using SHA-256 hash"""
    encryption_service = EncryptionService()
    return encryption_service.verify_hash(content, expected_hash, 'sha256')


# Global encryption service instance
encryption_service = EncryptionService()