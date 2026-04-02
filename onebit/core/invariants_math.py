"""Domain-Specific Invariants and Gates for Math/Code.

This module implements domain detection and adaptive compute allocation for
math and code domains. When math/code patterns are detected, compute budget (T)
is increased for surrounding tokens to improve accuracy.

Key features:
- Regex-based domain detection (math operators, code keywords)
- Balanced delimiter validation (parentheses, brackets, braces)
- Adaptive T allocation (boost T near domain triggers)
- Domain-specific error detection

Typical usage:
    # Detect domain
    detector = DomainDetector()
    domain = detector.detect("x = 2 + 3")  # Returns "math"
    
    # Get T boost
    gate = DomainGate(base_T=16, boost_T=24, window=2)
    tokens = ["x", "=", "2", "+", "3"]
    T_alloc = gate.allocate_T(tokens)  # [16, 24, 24, 24, 24]
    
    # Validate invariants
    validator = InvariantValidator()
    is_valid = validator.validate("(2 + 3) * 4")  # True
    is_valid = validator.validate("(2 + 3 * 4")    # False (unbalanced)
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from collections import Counter


# Domain detection patterns
# Math: operators, numbers, special symbols (but NOT single letters like 'e')
MATH_OPERATORS = r'[+\-*/=<>≤≥≠×÷^%]'
MATH_NUMBERS = r'\b\d+\.?\d*\b'
MATH_SYMBOLS = r'[π∞∑∏∫∂∇]'  # Removed 'e' and 'τ' to avoid false positives

# Code: keywords (with word boundaries), operators, special symbols
CODE_KEYWORDS = r'\b(def|class|if|else|elif|for|while|return|import|from|as|with|try|except|finally|raise|assert|break|continue|pass|yield|lambda|async|await)\b'
CODE_OPERATORS = r'(==|!=|<=|>=|&&|\|\||<<|>>|->|=>|\+=|-=|\*=|/=|%=|&=|\|=|\^=)'
CODE_SYMBOLS = r'[@#$&|~]'  # Removed '^' to avoid conflict with math

# Compile patterns
MATH_PATTERN = re.compile(f'({MATH_OPERATORS}|{MATH_NUMBERS}|{MATH_SYMBOLS})')
CODE_PATTERN = re.compile(f'({CODE_KEYWORDS}|{CODE_OPERATORS}|{CODE_SYMBOLS})')


@dataclass
class DomainTrigger:
    """Domain trigger information.
    
    Attributes:
        domain: Domain type ("math", "code", "text")
        position: Token position where trigger was detected
        pattern: Pattern that triggered detection
        confidence: Confidence score [0, 1]
    """
    domain: Literal["math", "code", "text"]
    position: int
    pattern: str
    confidence: float


class DomainDetector:
    """Detect domain (math/code/text) from token sequences."""

    def __init__(
        self,
        math_threshold: float = 0.3,
        code_threshold: float = 0.15,  # Lower threshold for code (keywords are sparse)
    ):
        """Initialize domain detector.

        Args:
            math_threshold: Minimum fraction of math tokens to trigger math domain
            code_threshold: Minimum fraction of code tokens to trigger code domain
        """
        self.math_threshold = math_threshold
        self.code_threshold = code_threshold
    
    def detect(self, text: str) -> Literal["math", "code", "text"]:
        """Detect domain from text.

        Args:
            text: Input text

        Returns:
            Domain type: "math", "code", or "text"
        """
        if not text.strip():
            return "text"

        # Count matches (code first to avoid double-counting)
        code_matches = len(CODE_PATTERN.findall(text))

        # Remove code matches from text before counting math
        text_no_code = CODE_PATTERN.sub('', text)
        math_matches = len(MATH_PATTERN.findall(text_no_code))

        # Normalize by text length
        text_len = len(text.split())
        if text_len == 0:
            return "text"

        math_density = math_matches / text_len
        code_density = code_matches / text_len

        # Decide domain (code takes priority)
        if code_density >= self.code_threshold:
            return "code"
        elif math_density >= self.math_threshold:
            return "math"
        else:
            return "text"
    
    def detect_triggers(self, tokens: List[str]) -> List[DomainTrigger]:
        """Detect domain triggers in token sequence.

        Args:
            tokens: List of tokens

        Returns:
            List of domain triggers
        """
        triggers = []

        for i, token in enumerate(tokens):
            # Check code patterns first (more specific)
            if CODE_PATTERN.search(token):
                code_matches = len(CODE_PATTERN.findall(token))
                confidence = min(1.0, code_matches / max(1, len(token)))
                triggers.append(DomainTrigger(
                    domain="code",
                    position=i,
                    pattern=token,
                    confidence=confidence,
                ))

            # Check math patterns (only if not code)
            elif MATH_PATTERN.search(token):
                math_matches = len(MATH_PATTERN.findall(token))
                confidence = min(1.0, math_matches / max(1, len(token)))
                triggers.append(DomainTrigger(
                    domain="math",
                    position=i,
                    pattern=token,
                    confidence=confidence,
                ))

        return triggers


class InvariantValidator:
    """Validate domain-specific invariants."""
    
    def validate(self, text: str) -> bool:
        """Validate text against domain invariants.
        
        Args:
            text: Input text
        
        Returns:
            True if all invariants are satisfied
        """
        # Check balanced delimiters
        if not self._check_balanced_delimiters(text):
            return False
        
        # Check valid operator sequences
        if not self._check_operator_sequences(text):
            return False
        
        return True
    
    def _check_balanced_delimiters(self, text: str) -> bool:
        """Check if delimiters are balanced.
        
        Args:
            text: Input text
        
        Returns:
            True if balanced
        """
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in text:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()
        
        return len(stack) == 0
    
    def _check_operator_sequences(self, text: str) -> bool:
        """Check for invalid operator sequences.
        
        Args:
            text: Input text
        
        Returns:
            True if valid
        """
        # Check for invalid sequences like "++", "--" (unless intentional)
        # For now, just check for triple operators
        invalid_patterns = [
            r'\+\+\+',
            r'---',
            r'\*\*\*',
            r'///',
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text):
                return False
        
        return True
    
    def get_violations(self, text: str) -> List[str]:
        """Get list of invariant violations.
        
        Args:
            text: Input text
        
        Returns:
            List of violation descriptions
        """
        violations = []
        
        if not self._check_balanced_delimiters(text):
            violations.append("Unbalanced delimiters")
        
        if not self._check_operator_sequences(text):
            violations.append("Invalid operator sequence")
        
        return violations


@dataclass
class DomainGateConfig:
    """Domain gate configuration.
    
    Attributes:
        base_T: Base T for text domain
        math_boost_T: T boost for math domain
        code_boost_T: T boost for code domain
        window: Number of tokens before/after trigger to boost
        decay: Decay factor for T boost (1.0 = no decay)
    """
    base_T: int = 16
    math_boost_T: int = 24
    code_boost_T: int = 24
    window: int = 2
    decay: float = 1.0


class DomainGate:
    """Adaptive T allocation based on domain triggers."""
    
    def __init__(self, cfg: DomainGateConfig):
        """Initialize domain gate.
        
        Args:
            cfg: Domain gate configuration
        """
        self.cfg = cfg
        self.detector = DomainDetector()
    
    def allocate_T(self, tokens: List[str]) -> List[int]:
        """Allocate T for each token based on domain triggers.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of T allocations (one per token)
        """
        n_tokens = len(tokens)
        T_alloc = [self.cfg.base_T] * n_tokens
        
        # Detect triggers
        triggers = self.detector.detect_triggers(tokens)
        
        # Boost T around triggers
        for trigger in triggers:
            pos = trigger.position
            
            # Determine boost amount
            if trigger.domain == "math":
                boost_T = self.cfg.math_boost_T
            elif trigger.domain == "code":
                boost_T = self.cfg.code_boost_T
            else:
                continue
            
            # Apply boost to window
            for offset in range(-self.cfg.window, self.cfg.window + 1):
                idx = pos + offset
                if 0 <= idx < n_tokens:
                    # Apply decay
                    distance = abs(offset)
                    decay_factor = self.cfg.decay ** distance
                    boosted_T = int(self.cfg.base_T + (boost_T - self.cfg.base_T) * decay_factor)
                    
                    # Take maximum (multiple triggers can overlap)
                    T_alloc[idx] = max(T_alloc[idx], boosted_T)
        
        return T_alloc
    
    def get_allocation_summary(self, tokens: List[str]) -> Dict:
        """Get summary of T allocation.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Dictionary with allocation statistics
        """
        T_alloc = self.allocate_T(tokens)
        triggers = self.detector.detect_triggers(tokens)
        
        # Count domains
        domain_counts = Counter(t.domain for t in triggers)
        
        return {
            "n_tokens": len(tokens),
            "n_triggers": len(triggers),
            "n_math": domain_counts.get("math", 0),
            "n_code": domain_counts.get("code", 0),
            "T_mean": sum(T_alloc) / len(T_alloc) if T_alloc else 0,
            "T_min": min(T_alloc) if T_alloc else 0,
            "T_max": max(T_alloc) if T_alloc else 0,
            "total_compute": sum(T_alloc),
        }

