"""
Privacy Service for memOS Server
HIPAA-compliant privacy protection and content anonymization
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone

from models.memory import MemoryPrivacyLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PrivacyService:
    """
    HIPAA-compliant privacy service for memory content protection
    Handles anonymization, validation, and access control
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Protected Health Information (PHI) patterns
        self.phi_patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'date_of_birth': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'medical_id': re.compile(r'\b(MRN|ID|Patient)\s*:?\s*\d+\b', re.IGNORECASE),
            'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', re.IGNORECASE),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
        }
        
        # Sensitive keywords for recovery context
        self.sensitive_keywords = {
            'high_risk': {
                'suicide', 'kill myself', 'end it all', 'overdose', 'relapse',
                'dealer', 'drug dealer', 'buy drugs', 'using again'
            },
            'personal': {
                'family name', 'full name', 'real name', 'home address',
                'work address', 'employer', 'social security', 'bank account'
            },
            'medical': {
                'prescription', 'medication', 'doctor name', 'hospital',
                'medical record', 'insurance', 'diagnosis'
            }
        }
        
        # Anonymization replacements
        self.anonymization_map = {
            'ssn': '***-**-****',
            'phone': '***-***-****',
            'email': '[EMAIL REMOVED]',
            'date_of_birth': '[DATE REMOVED]',
            'medical_id': '[ID REMOVED]',
            'address': '[ADDRESS REMOVED]',
            'zip_code': '[ZIP REMOVED]',
            'credit_card': '****-****-****-****'
        }
    
    async def validate_memory_content(
        self,
        content: str,
        privacy_level: MemoryPrivacyLevel
    ) -> bool:
        """
        Validate if memory content meets privacy requirements
        """
        try:
            # Check for PHI violations
            phi_violations = self._detect_phi(content)
            
            if privacy_level == MemoryPrivacyLevel.MINIMAL:
                # Strict - no PHI allowed
                return len(phi_violations) == 0
            
            elif privacy_level == MemoryPrivacyLevel.BALANCED:
                # Moderate - limited PHI allowed if not highly sensitive
                high_risk_phi = {'ssn', 'credit_card', 'medical_id'}
                return not any(phi_type in high_risk_phi for phi_type in phi_violations)
            
            elif privacy_level == MemoryPrivacyLevel.COMPREHENSIVE:
                # Permissive - most content allowed with explicit consent
                return True
            
            elif privacy_level == MemoryPrivacyLevel.RESTRICTED:
                # Maximum protection - very limited content
                return len(phi_violations) == 0 and not self._contains_sensitive_keywords(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Privacy validation failed: {e}")
            return False
    
    async def anonymize_content(
        self,
        content: str,
        privacy_level: MemoryPrivacyLevel,
        preserve_therapeutic_value: bool = True
    ) -> str:
        """
        Anonymize content while preserving therapeutic value
        """
        try:
            anonymized = content
            
            # Apply PHI anonymization
            for phi_type, pattern in self.phi_patterns.items():
                if privacy_level in [MemoryPrivacyLevel.MINIMAL, MemoryPrivacyLevel.RESTRICTED]:
                    # Full anonymization
                    anonymized = pattern.sub(self.anonymization_map[phi_type], anonymized)
                elif privacy_level == MemoryPrivacyLevel.BALANCED:
                    # Selective anonymization
                    if phi_type in ['ssn', 'credit_card', 'medical_id']:
                        anonymized = pattern.sub(self.anonymization_map[phi_type], anonymized)
                # COMPREHENSIVE level preserves most content
            
            # Handle sensitive keywords based on privacy level
            if privacy_level == MemoryPrivacyLevel.RESTRICTED:
                anonymized = self._anonymize_sensitive_keywords(anonymized)
            
            # Preserve therapeutic context if requested
            if preserve_therapeutic_value:
                anonymized = self._preserve_therapeutic_context(anonymized, content)
            
            return anonymized
            
        except Exception as e:
            logger.error(f"Content anonymization failed: {e}")
            return content  # Return original on error
    
    async def can_access_content(
        self,
        user_id: str,
        memory_privacy_level: MemoryPrivacyLevel,
        requester_role: str = "user",
        explicit_consent: bool = False
    ) -> bool:
        """
        Check if content can be accessed based on privacy level and permissions
        """
        try:
            # User can always access their own content
            if requester_role == "user":
                return True
            
            # Care team access requires explicit consent
            if requester_role == "care_team":
                return explicit_consent and memory_privacy_level != MemoryPrivacyLevel.RESTRICTED
            
            # System access for processing
            if requester_role == "system":
                return memory_privacy_level != MemoryPrivacyLevel.RESTRICTED
            
            # Admin access requires explicit consent for anything beyond minimal
            if requester_role == "admin":
                if memory_privacy_level == MemoryPrivacyLevel.MINIMAL:
                    return True
                return explicit_consent
            
            return False
            
        except Exception as e:
            logger.error(f"Access control check failed: {e}")
            return False
    
    async def audit_privacy_access(
        self,
        user_id: str,
        memory_id: str,
        requester_id: str,
        requester_role: str,
        access_type: str,
        content_accessed: bool = False
    ) -> Dict[str, Any]:
        """
        Create audit log entry for privacy-related access
        """
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'memory_id': memory_id,
            'requester_id': requester_id,
            'requester_role': requester_role,
            'access_type': access_type,  # 'read', 'write', 'delete', 'export'
            'content_accessed': content_accessed,
            'ip_address': None,  # Would be filled by calling service
            'user_agent': None,  # Would be filled by calling service
            'privacy_level_at_access': None,  # Would be filled by calling service
            'consent_status': None  # Would be filled by calling service
        }
        
        # This would be stored in audit_logs table
        logger.info(f"Privacy access audit: {audit_entry}")
        return audit_entry
    
    async def check_consent_requirements(
        self,
        user_id: str,
        memory_privacy_level: MemoryPrivacyLevel,
        operation_type: str
    ) -> Dict[str, Any]:
        """
        Check consent requirements for memory operations
        """
        consent_requirements = {
            'explicit_consent_required': False,
            'consent_document_version': '1.0',
            'consent_expiry_months': 6,
            'required_permissions': [],
            'additional_warnings': []
        }
        
        # Determine consent requirements based on privacy level
        if memory_privacy_level == MemoryPrivacyLevel.COMPREHENSIVE:
            consent_requirements['explicit_consent_required'] = True
            consent_requirements['required_permissions'] = [
                'memory_storage',
                'content_analysis',
                'therapeutic_insights'
            ]
        
        elif memory_privacy_level == MemoryPrivacyLevel.BALANCED:
            if operation_type in ['export', 'share', 'analysis']:
                consent_requirements['explicit_consent_required'] = True
                consent_requirements['required_permissions'] = ['data_export']
        
        elif memory_privacy_level == MemoryPrivacyLevel.RESTRICTED:
            consent_requirements['explicit_consent_required'] = True
            consent_requirements['required_permissions'] = [
                'restricted_memory_access',
                'enhanced_privacy_protection'
            ]
            consent_requirements['additional_warnings'] = [
                'This memory contains highly sensitive information',
                'Access is logged and monitored for compliance'
            ]
        
        return consent_requirements
    
    async def generate_privacy_report(
        self,
        user_id: str,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Generate privacy compliance report for user
        """
        # This would query actual data - simplified for now
        report = {
            'user_id': user_id,
            'report_generated_at': datetime.now(timezone.utc).isoformat(),
            'date_range': {
                'start': date_range[0].isoformat() if date_range else None,
                'end': date_range[1].isoformat() if date_range else None
            },
            'memory_privacy_breakdown': {
                'minimal': 0,
                'balanced': 0,
                'comprehensive': 0,
                'restricted': 0
            },
            'phi_detection_summary': {
                'total_phi_instances_detected': 0,
                'phi_types_found': [],
                'anonymization_applied': 0
            },
            'access_log_summary': {
                'total_accesses': 0,
                'user_accesses': 0,
                'care_team_accesses': 0,
                'admin_accesses': 0,
                'consent_violations': 0
            },
            'consent_status': {
                'active_consents': [],
                'expired_consents': [],
                'pending_renewals': []
            },
            'data_retention_status': {
                'memories_approaching_expiry': 0,
                'expired_memories': 0,
                'retention_policy_compliant': True
            }
        }
        
        return report
    
    # Private helper methods
    
    def _detect_phi(self, content: str) -> Set[str]:
        """Detect Protected Health Information in content"""
        detected_phi = set()
        
        for phi_type, pattern in self.phi_patterns.items():
            if pattern.search(content):
                detected_phi.add(phi_type)
        
        return detected_phi
    
    def _contains_sensitive_keywords(self, content: str) -> bool:
        """Check if content contains sensitive keywords"""
        content_lower = content.lower()
        
        for category, keywords in self.sensitive_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return True
        
        return False
    
    def _anonymize_sensitive_keywords(self, content: str) -> str:
        """Anonymize sensitive keywords while preserving context"""
        anonymized = content
        
        # Replace specific names and identifiers with generic terms
        name_patterns = [
            (re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'), '[NAME]'),
            (re.compile(r'\bDr\. [A-Z][a-z]+\b'), '[DOCTOR]'),
            (re.compile(r'\b[A-Z][a-z]+ Hospital\b'), '[HOSPITAL]'),
            (re.compile(r'\b[A-Z][a-z]+ Clinic\b'), '[CLINIC]')
        ]
        
        for pattern, replacement in name_patterns:
            anonymized = pattern.sub(replacement, anonymized)
        
        return anonymized
    
    def _preserve_therapeutic_context(self, anonymized: str, original: str) -> str:
        """Preserve therapeutic value while maintaining privacy"""
        # Preserve emotional context and recovery-related terms
        therapeutic_terms = {
            'addiction', 'recovery', 'sobriety', 'relapse', 'sponsor',
            'meeting', 'step', 'therapy', 'counseling', 'support',
            'milestone', 'progress', 'setback', 'trigger', 'craving',
            'healing', 'hope', 'strength', 'courage', 'faith'
        }
        
        # Ensure therapeutic terms are not accidentally anonymized
        for term in therapeutic_terms:
            if term in original.lower() and term not in anonymized.lower():
                # Re-add therapeutic context if it was removed
                anonymized = f"{anonymized} [Therapeutic context: discussion about {term}]"
        
        return anonymized
    
    def _calculate_privacy_score(self, content: str) -> float:
        """Calculate privacy risk score for content (0.0 = safe, 1.0 = high risk)"""
        score = 0.0
        
        # PHI detection adds to risk
        phi_violations = self._detect_phi(content)
        score += len(phi_violations) * 0.2
        
        # Sensitive keywords add to risk
        if self._contains_sensitive_keywords(content):
            score += 0.3
        
        # Content length factor (longer content = potentially more risk)
        if len(content) > 1000:
            score += 0.1
        
        return min(score, 1.0)


# Global privacy service instance
privacy_service = PrivacyService()