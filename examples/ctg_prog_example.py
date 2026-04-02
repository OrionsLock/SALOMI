"""Example: Using CTG-PROG v1 with adaptive program selection.

This example demonstrates:
1. Creating an adaptive program selector
2. Extracting features from runtime state
3. Selecting a program dynamically
4. Running CTG-PROG inference
5. Training the selector with composite loss
"""
from __future__ import annotations

import numpy as np
import torch

from onebit.runtime.ctg_grammar import CTG, CTGState, make_default_programs
from onebit.runtime.ctg_selector import (
    AdaptiveProgramSelector,
    SelectorConfig,
    extract_features,
)
from onebit.training.ctg_trainer import CTGTrainer, TrainingConfig
from onebit.ops.logits_sprt import shortlist_and_certify
from onebit.core.packbits import pack_input_signs


def example_inference():
    """Example: Inference with adaptive program selection."""
    print("="*60)
    print("CTG-PROG Inference Example")
    print("="*60)
    
    # 1. Create CTG with K=4 programs
    vocab_size = 1000
    programs = make_default_programs(vocab_size, K=4)
    ctg = CTG(programs=programs, vocab_size=vocab_size, phase_period=8)
    ctg_state = CTGState()
    
    # 2. Create adaptive selector
    selector_cfg = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(selector_cfg)
    selector.eval()  # Inference mode
    
    # 3. Simulate runtime features
    shortlist_logits = np.array([2.5, 1.8, 1.2, 0.9, 0.5], dtype=np.float32)
    top2_margin = 0.7
    attn_entropy = 1.5
    phase_history = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    
    # 4. Extract features
    features = extract_features(
        shortlist_logits, top2_margin, attn_entropy, phase_history
    )
    print(f"\nExtracted features (8D): {features}")
    
    # 5. Select program
    features_t = torch.tensor(features).unsqueeze(0)
    probs, program_id = selector(features_t, tau=0.2, hard=True)
    program_id = int(program_id.item())
    print(f"Selected program: {program_id}")
    print(f"Program probabilities: {probs.squeeze().detach().numpy()}")
    
    # 6. Run CTG-PROG inference
    rng = np.random.default_rng(42)
    d = 256
    d_words = d // 32
    
    q_bits = rng.integers(0, 2**32 - 1, size=(d_words,), dtype=np.uint32)
    v_ids = np.arange(vocab_size, dtype=np.int32)
    
    result = shortlist_and_certify(
        q_bits, v_ids,
        d=d,
        k0=8,
        k_step=4,
        k_max=32,
        shortlist_size=16,
        eps=0.05,
        delta=0.01,
        backend="cpu",
        prf_seed=12345,
        use_ctg=0,
        ctg=ctg,
        ctg_state=ctg_state,
        ctg_program_id=program_id,
    )
    
    print(f"\nInference result:")
    print(f"  k_used: {result['k_used']}")
    print(f"  pairs_evaluated: {result.get('pairs_evaluated', 0)}")
    print(f"  ctg_prog_id: {result['ctg_prog_id']}")
    print(f"  ctg_phase: {result['ctg_phase']}")
    print(f"  ctg_masked_count: {result['ctg_masked_count']}")


def example_training():
    """Example: Training the adaptive selector."""
    print("\n" + "="*60)
    print("CTG-PROG Training Example")
    print("="*60)
    
    # 1. Create selector and trainer
    selector_cfg = SelectorConfig(K=4, hidden_dim=32, feature_dim=8)
    selector = AdaptiveProgramSelector(selector_cfg)
    
    trainer_cfg = TrainingConfig(
        lambda_var=0.1,
        lambda_switch=0.05,
        lambda_entropy=0.01,
        learning_rate=1e-3,
        max_steps=100,
        log_interval=20,
    )
    trainer = CTGTrainer(selector, trainer_cfg, device="cpu")
    
    # 2. Simulate training loop
    print("\nTraining for 100 steps...")
    
    for step in range(100):
        # Simulate batch of features
        batch_size = 8
        features = torch.randn(batch_size, 8)
        
        # Simulate base loss and k variance
        base_loss = torch.tensor(2.5 - step * 0.01)  # Decreasing loss
        k_variance = 10.0 - step * 0.05  # Decreasing variance
        
        # Anneal temperature
        tau = 1.0 - (step / 100) * 0.8  # 1.0 → 0.2
        
        # Training step
        losses = trainer.train_step(features, base_loss, k_variance, tau)
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Total loss: {losses['total']:.4f}")
            print(f"  Base loss: {losses['base']:.4f}")
            print(f"  Var loss: {losses['var']:.4f}")
            print(f"  Switch loss: {losses['switch']:.4f}")
            print(f"  Entropy loss: {losses['entropy']:.4f}")
    
    # 3. Check program usage
    usage = trainer.get_program_usage_histogram()
    print(f"\nProgram usage histogram:")
    for prog_id, frac in usage.items():
        print(f"  Program {prog_id}: {frac:.2%}")
    
    # Verify all programs are active
    min_usage = min(usage.values())
    if min_usage > 0.05:
        print("\n✅ All programs remain active (>5% usage)")
    else:
        print(f"\n⚠️  Warning: Some programs underused (min: {min_usage:.2%})")


def example_comparison():
    """Example: Compare CTG-FIXED vs CTG-PROG."""
    print("\n" + "="*60)
    print("CTG-FIXED vs CTG-PROG Comparison")
    print("="*60)
    
    vocab_size = 500
    rng = np.random.default_rng(123)
    d = 256
    d_words = d // 32
    
    q_bits = rng.integers(0, 2**32 - 1, size=(d_words,), dtype=np.uint32)
    v_ids = np.arange(vocab_size, dtype=np.int32)
    
    kwargs = dict(
        d=d, k0=8, k_step=4, k_max=32, shortlist_size=16,
        eps=0.05, delta=0.01, backend="cpu", prf_seed=12345, use_ctg=0,
    )
    
    # CTG-FIXED
    from onebit.runtime.ctg_grammar import CTGRule
    rules = [CTGRule(op="PASS", ids=None)]
    ctg_fixed = CTG(rules=rules, vocab_size=vocab_size)
    result_fixed = shortlist_and_certify(
        q_bits, v_ids, ctg=ctg_fixed, ctg_state=CTGState(), ctg_program_id=0, **kwargs
    )
    
    # CTG-PROG (program 2: INHIBIT-spiky)
    programs = make_default_programs(vocab_size, K=4)
    ctg_prog = CTG(programs=programs, vocab_size=vocab_size)
    result_prog = shortlist_and_certify(
        q_bits, v_ids, ctg=ctg_prog, ctg_state=CTGState(), ctg_program_id=2, **kwargs
    )
    
    print(f"\nCTG-FIXED:")
    print(f"  pairs_evaluated: {result_fixed.get('pairs_evaluated', 0)}")
    print(f"  k_used: {result_fixed['k_used']}")
    
    print(f"\nCTG-PROG (program 2):")
    print(f"  pairs_evaluated: {result_prog.get('pairs_evaluated', 0)}")
    print(f"  k_used: {result_prog['k_used']}")
    
    pairs_reduction = (
        (result_fixed.get('pairs_evaluated', 0) - result_prog.get('pairs_evaluated', 0))
        / result_fixed.get('pairs_evaluated', 1)
    ) * 100
    print(f"\nPairs reduction: {pairs_reduction:.1f}%")


if __name__ == "__main__":
    example_inference()
    example_training()
    example_comparison()
    
    print("\n" + "="*60)
    print("✅ All examples completed successfully!")
    print("="*60)

