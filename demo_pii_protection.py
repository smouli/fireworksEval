#!/usr/bin/env python3
"""
Demonstrate Fireworks PII Protection features.
Shows zero data retention, encryption, and compliance.
"""

import os
import re
from typing import Dict, List

class PIIProtectionDemo:
    def detect_pii(self, text: str) -> List[str]:
        """Simple PII detection (for demo purposes)."""
        pii_types = []
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            pii_types.append("email")
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            pii_types.append("phone")
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            pii_types.append("ssn")
        return pii_types
    
    def analyze_pii_protection(self, question: str) -> Dict:
        """Analyze how Fireworks protects PII in this query."""
        pii_types = self.detect_pii(question)
        return {
            "pii_detected": pii_types,
            "zero_data_retention": True,
            "encryption": "TLS in transit, AES at rest",
            "compliance": "SOC 2 Type II, HIPAA"
        }
    
    def demo_pii_protection(self):
        """Demonstrate PII protection features."""
        print("=" * 80)
        print("Fireworks PII Protection Demo")
        print("=" * 80)
        
        test_cases = [
            "Show trips for customer john.doe@email.com",
            "Find driver with phone number 555-123-4567",
            "List all trips for customer ID 12345 in Seattle",
        ]
        
        print("\n1. Zero Data Retention Policy:")
        print("-" * 80)
        print("   ✓ No logging of prompts or generations (default)")
        print("   ✓ Data exists only in volatile memory during request")
        print("   ✓ No persistent storage of sensitive information")
        print("   ✓ Compliant with GDPR, HIPAA, SOC 2 Type II")
        
        print("\n2. Encryption:")
        print("-" * 80)
        print("   ✓ All data encrypted in transit (TLS)")
        print("   ✓ All data encrypted at rest")
        print("   ✓ Industry-standard encryption protocols")
        
        print("\n3. Example Queries with PII:")
        print("-" * 80)
        for i, question in enumerate(test_cases, 1):
            result = self.analyze_pii_protection(question)
            print(f"\n   Query {i}: {question}")
            print(f"   PII Detected: {', '.join(result['pii_detected']) if result['pii_detected'] else 'None'}")
            print(f"   Protection: Zero Data Retention (default)")
            print(f"   Compliance: {result['compliance']}")
        
        print("\n" + "=" * 80)
        print("Key Takeaways:")
        print("  - Zero data retention by default (no opt-in required)")
        print("  - SOC 2 Type II and HIPAA compliant")
        print("  - Encryption at rest and in transit")
        print("  - Suitable for healthcare, finance, and regulated industries")

if __name__ == "__main__":
    demo = PIIProtectionDemo()
    demo.demo_pii_protection()

