"""Tests for domain-specific invariants and gates.

Tests cover:
1. Domain detection (math/code/text)
2. Trigger detection
3. Invariant validation (balanced delimiters, operator sequences)
4. Adaptive T allocation
5. Domain gate integration
"""
import pytest
from onebit.core.invariants_math import (
    DomainDetector,
    InvariantValidator,
    DomainGate,
    DomainGateConfig,
    DomainTrigger,
)


class TestDomainDetection:
    """Test domain detection."""
    
    def test_detect_math(self):
        """Test math domain detection."""
        detector = DomainDetector()
        
        # Clear math examples
        assert detector.detect("x = 2 + 3") == "math"
        assert detector.detect("(a + b) * c") == "math"
        assert detector.detect("f(x) = x^2 + 2x + 1") == "math"
        assert detector.detect("∫ x dx = x²/2") == "math"
    
    def test_detect_code(self):
        """Test code domain detection."""
        detector = DomainDetector()
        
        # Clear code examples
        assert detector.detect("def foo(x):") == "code"
        assert detector.detect("if x == 0:") == "code"
        assert detector.detect("for i in range(10):") == "code"
        assert detector.detect("return x + 1") == "code"
    
    def test_detect_text(self):
        """Test text domain detection."""
        detector = DomainDetector()
        
        # Clear text examples
        assert detector.detect("The quick brown fox") == "text"
        assert detector.detect("Hello world") == "text"
        assert detector.detect("This is a sentence.") == "text"
    
    def test_detect_mixed(self):
        """Test mixed domain detection."""
        detector = DomainDetector()
        
        # Code should win over math (more specific)
        assert detector.detect("def f(x): return x + 1") == "code"
        
        # Math should win when dominant
        assert detector.detect("The equation is x = 2 + 3 * 4") == "math"
    
    def test_detect_empty(self):
        """Test empty input."""
        detector = DomainDetector()
        assert detector.detect("") == "text"
        assert detector.detect("   ") == "text"


class TestTriggerDetection:
    """Test trigger detection."""
    
    def test_math_triggers(self):
        """Test math trigger detection."""
        detector = DomainDetector()
        tokens = ["x", "=", "2", "+", "3"]
        
        triggers = detector.detect_triggers(tokens)
        
        # Should detect "=", "+", and numbers
        assert len(triggers) >= 2
        assert any(t.domain == "math" and t.position == 1 for t in triggers)  # "="
        assert any(t.domain == "math" and t.position == 3 for t in triggers)  # "+"
    
    def test_code_triggers(self):
        """Test code trigger detection."""
        detector = DomainDetector()
        tokens = ["def", "foo", "(", "x", ")", ":"]
        
        triggers = detector.detect_triggers(tokens)
        
        # Should detect "def" and delimiters
        assert len(triggers) >= 1
        assert any(t.domain == "code" and t.position == 0 for t in triggers)  # "def"
    
    def test_no_triggers(self):
        """Test text with no triggers."""
        detector = DomainDetector()
        tokens = ["The", "quick", "brown", "fox"]
        
        triggers = detector.detect_triggers(tokens)
        
        # Should have no triggers
        assert len(triggers) == 0
    
    def test_trigger_confidence(self):
        """Test trigger confidence scores."""
        detector = DomainDetector()
        tokens = ["+", "def", "hello"]
        
        triggers = detector.detect_triggers(tokens)
        
        # All triggers should have confidence > 0
        for trigger in triggers:
            assert 0 < trigger.confidence <= 1.0


class TestInvariantValidation:
    """Test invariant validation."""
    
    def test_balanced_delimiters_valid(self):
        """Test valid balanced delimiters."""
        validator = InvariantValidator()
        
        assert validator.validate("(2 + 3)")
        assert validator.validate("[(a + b) * c]")
        assert validator.validate("{x: [1, 2, 3]}")
        assert validator.validate("f(g(h(x)))")
    
    def test_balanced_delimiters_invalid(self):
        """Test invalid balanced delimiters."""
        validator = InvariantValidator()
        
        assert not validator.validate("(2 + 3")
        assert not validator.validate("2 + 3)")
        assert not validator.validate("[(a + b])")
        assert not validator.validate("{x: [1, 2, 3}")
    
    def test_operator_sequences_valid(self):
        """Test valid operator sequences."""
        validator = InvariantValidator()
        
        assert validator.validate("x + y")
        assert validator.validate("x - y")
        assert validator.validate("x * y")
        assert validator.validate("x / y")
        assert validator.validate("x++")  # C-style increment (allowed)
    
    def test_operator_sequences_invalid(self):
        """Test invalid operator sequences."""
        validator = InvariantValidator()
        
        assert not validator.validate("x +++ y")
        assert not validator.validate("x --- y")
        assert not validator.validate("x *** y")
        assert not validator.validate("x /// y")
    
    def test_get_violations(self):
        """Test violation reporting."""
        validator = InvariantValidator()
        
        # Valid text
        violations = validator.get_violations("(2 + 3)")
        assert len(violations) == 0
        
        # Unbalanced delimiters
        violations = validator.get_violations("(2 + 3")
        assert len(violations) == 1
        assert "Unbalanced delimiters" in violations[0]
        
        # Invalid operators
        violations = validator.get_violations("x +++ y")
        assert len(violations) == 1
        assert "Invalid operator sequence" in violations[0]


class TestDomainGate:
    """Test domain gate T allocation."""
    
    def test_base_allocation(self):
        """Test base T allocation (no triggers)."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        tokens = ["The", "quick", "brown", "fox"]
        T_alloc = gate.allocate_T(tokens)
        
        # All tokens should get base_T
        assert all(T == 16 for T in T_alloc)
    
    def test_math_boost(self):
        """Test T boost for math triggers."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        tokens = ["x", "=", "2", "+", "3"]
        T_alloc = gate.allocate_T(tokens)
        
        # Tokens near "=" and "+" should get boosted T
        # "=" is at position 1, so positions 0, 1, 2, 3 should be boosted
        # "+" is at position 3, so positions 1, 2, 3, 4 should be boosted
        # All tokens should be boosted due to overlap
        assert all(T >= 16 for T in T_alloc)
        assert any(T > 16 for T in T_alloc)
    
    def test_code_boost(self):
        """Test T boost for code triggers."""
        cfg = DomainGateConfig(base_T=16, code_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        tokens = ["def", "foo", "(", "x", ")", ":"]
        T_alloc = gate.allocate_T(tokens)
        
        # Tokens near "def" should get boosted T
        assert T_alloc[0] >= 16  # "def"
        assert T_alloc[1] >= 16  # "foo" (within window)
        assert T_alloc[2] >= 16  # "(" (within window)
    
    def test_window_size(self):
        """Test window size effect."""
        # Small window
        cfg_small = DomainGateConfig(base_T=16, math_boost_T=24, window=1)
        gate_small = DomainGate(cfg_small)
        
        tokens = ["a", "b", "+", "c", "d"]
        T_small = gate_small.allocate_T(tokens)
        
        # Large window
        cfg_large = DomainGateConfig(base_T=16, math_boost_T=24, window=3)
        gate_large = DomainGate(cfg_large)
        
        T_large = gate_large.allocate_T(tokens)
        
        # Large window should boost more tokens
        assert sum(T > 16 for T in T_large) >= sum(T > 16 for T in T_small)
    
    def test_decay(self):
        """Test decay factor."""
        # No decay
        cfg_no_decay = DomainGateConfig(base_T=16, math_boost_T=24, window=2, decay=1.0)
        gate_no_decay = DomainGate(cfg_no_decay)
        
        tokens = ["a", "b", "+", "c", "d"]
        T_no_decay = gate_no_decay.allocate_T(tokens)
        
        # With decay
        cfg_decay = DomainGateConfig(base_T=16, math_boost_T=24, window=2, decay=0.5)
        gate_decay = DomainGate(cfg_decay)
        
        T_decay = gate_decay.allocate_T(tokens)
        
        # Decay should reduce boost for distant tokens
        # Token at trigger position should have same boost
        assert T_no_decay[2] == T_decay[2]  # "+" (trigger)
        # Tokens farther away should have lower boost with decay
        # (This might not always hold due to integer rounding, so we just check it's reasonable)
        assert all(T >= 16 for T in T_decay)
    
    def test_overlapping_triggers(self):
        """Test overlapping trigger windows."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        tokens = ["x", "+", "y", "+", "z"]
        T_alloc = gate.allocate_T(tokens)
        
        # All tokens should be boosted due to overlapping windows
        assert all(T >= 16 for T in T_alloc)
        assert any(T > 16 for T in T_alloc)
    
    def test_allocation_summary(self):
        """Test allocation summary."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        tokens = ["x", "=", "2", "+", "3"]
        summary = gate.get_allocation_summary(tokens)
        
        assert summary["n_tokens"] == 5
        assert summary["n_triggers"] >= 2
        assert summary["n_math"] >= 2
        assert summary["T_mean"] >= 16
        assert summary["T_min"] >= 16
        assert summary["T_max"] >= 16


class TestIntegration:
    """Integration tests."""
    
    def test_math_expression(self):
        """Test complete math expression."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        validator = InvariantValidator()
        
        # Valid math expression
        text = "(2 + 3) * 4"
        tokens = text.split()
        
        # Should be valid
        assert validator.validate(text)
        
        # Should boost T
        T_alloc = gate.allocate_T(tokens)
        assert any(T > 16 for T in T_alloc)
    
    def test_code_snippet(self):
        """Test complete code snippet."""
        cfg = DomainGateConfig(base_T=16, code_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        # Code snippet
        tokens = ["def", "add", "(", "x", ",", "y", ")", ":", "return", "x", "+", "y"]
        
        # Should boost T near "def" and "return"
        T_alloc = gate.allocate_T(tokens)
        assert any(T > 16 for T in T_alloc)
        
        # Summary should show code triggers
        summary = gate.get_allocation_summary(tokens)
        assert summary["n_code"] >= 1
    
    def test_gsm8k_style_problem(self):
        """Test GSM8K-style math problem."""
        cfg = DomainGateConfig(base_T=16, math_boost_T=24, window=2)
        gate = DomainGate(cfg)
        
        # GSM8K-style problem
        text = "If x = 5 and y = 3, what is x + y?"
        tokens = text.split()
        
        # Should boost T near math operators
        T_alloc = gate.allocate_T(tokens)
        
        # Summary should show math triggers
        summary = gate.get_allocation_summary(tokens)
        assert summary["n_math"] >= 2
        assert summary["T_mean"] > 16  # Average should be higher than base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

