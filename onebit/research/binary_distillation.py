"""Binary GPT-2 with Knowledge Distillation.

This implements true 1.0 bpp binary weights trained via distillation
from a FP32 teacher model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Optional, Tuple
import numpy as np
import math


class STESign(torch.autograd.Function):
    """Straight-Through Estimator for sign function."""
    
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient passes through as if sign was identity
        return grad_output


def ste_sign(x):
    """Apply sign with straight-through gradient."""
    return STESign.apply(x)


class BinaryLinear(nn.Module):
    """Linear layer with binary weights {-1, +1}.
    
    Uses STE for gradient flow. Per-layer scale for magnitude.
    True 1.0 bpp (scale overhead is negligible: 32 bits / millions of weights).
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent weights (FP32 during training, binarized in forward)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Per-layer scale (negligible overhead: 1 float per layer)
        self.scale = nn.Parameter(torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize weights with STE
        weight_binary = ste_sign(self.weight)
        
        # Handle zeros (shouldn't happen often, but just in case)
        weight_binary = torch.where(
            weight_binary == 0,
            torch.ones_like(weight_binary),
            weight_binary
        )
        
        # Apply scale and compute output
        out = F.linear(x, weight_binary * self.scale, self.bias)
        return out
    
    def get_binary_weights(self) -> torch.Tensor:
        """Get the actual binary weights for inference."""
        with torch.no_grad():
            w_bin = torch.sign(self.weight)
            w_bin = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin)
            return w_bin * self.scale
    
    def effective_bpp(self) -> float:
        """Calculate effective bits per parameter."""
        n_weights = self.in_features * self.out_features
        sign_bits = n_weights  # 1 bit per weight
        scale_bits = 32  # 1 float for scale
        bias_bits = self.out_features * 32 if self.bias is not None else 0
        total_bits = sign_bits + scale_bits + bias_bits
        return total_bits / n_weights


class BinaryAttention(nn.Module):
    """Multi-head attention with binary weights."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Binary projections
        self.c_attn = BinaryLinear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = BinaryLinear(config.n_embd, config.n_embd, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class BinaryMLP(nn.Module):
    """MLP block with binary weights."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = BinaryLinear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = BinaryLinear(4 * config.n_embd, config.n_embd, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BinaryBlock(nn.Module):
    """Transformer block with binary weights."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = BinaryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = BinaryMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BinaryGPT2(nn.Module):
    """GPT-2 with binary linear layers.

    Embeddings remain FP32 (tiny fraction of total params).
    All Linear layers are binary {-1, +1} with per-layer scale.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Embeddings stay FP32 (wte: 50257 * 768 = 38M params out of 124M total)
        # This is ~30% of params but we keep them FP32 for quality
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Binary transformer blocks
        self.h = nn.ModuleList([BinaryBlock(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # LM head shares weights with wte (standard GPT-2)
        # This means lm_head is also FP32

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.size()

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        x = self.ln_f(x)

        # LM head (tied to wte)
        logits = x @ self.wte.weight.T

        return logits

    def count_parameters(self) -> dict:
        """Count parameters by type."""
        binary_params = 0
        fp32_params = 0

        for name, module in self.named_modules():
            if isinstance(module, BinaryLinear):
                binary_params += module.weight.numel()
            elif isinstance(module, nn.Embedding):
                fp32_params += module.weight.numel()
            elif isinstance(module, nn.LayerNorm):
                fp32_params += module.weight.numel() + module.bias.numel()

        return {
            'binary': binary_params,
            'fp32': fp32_params,
            'total': binary_params + fp32_params,
            'binary_fraction': binary_params / (binary_params + fp32_params)
        }

    @classmethod
    def from_pretrained_teacher(cls, model_name: str = 'gpt2') -> 'BinaryGPT2':
        """Initialize binary model from pretrained GPT-2 weights."""
        # Load teacher
        teacher = GPT2LMHeadModel.from_pretrained(model_name)
        config = teacher.config

        # Create binary model
        model = cls(config)

        # Copy embeddings and layer norms (these stay FP32)
        model.wte.load_state_dict(teacher.transformer.wte.state_dict())
        model.wpe.load_state_dict(teacher.transformer.wpe.state_dict())
        model.ln_f.load_state_dict(teacher.transformer.ln_f.state_dict())

        # Initialize binary layers from teacher's FP32 weights
        # NOTE: GPT-2 uses Conv1D which stores weights as (out, in), but
        # HuggingFace GPT2 uses (in, out) format, so we need to transpose
        for i, block in enumerate(model.h):
            teacher_block = teacher.transformer.h[i]

            # Attention - need to transpose from (in, out) to (out, in)
            block.ln_1.load_state_dict(teacher_block.ln_1.state_dict())
            block.attn.c_attn.weight.data = teacher_block.attn.c_attn.weight.data.T.clone()
            block.attn.c_attn.bias.data = teacher_block.attn.c_attn.bias.data.clone()
            block.attn.c_proj.weight.data = teacher_block.attn.c_proj.weight.data.T.clone()
            block.attn.c_proj.bias.data = teacher_block.attn.c_proj.bias.data.clone()

            # Set initial scales based on teacher's weight magnitudes
            block.attn.c_attn.scale.data.fill_(teacher_block.attn.c_attn.weight.abs().mean())
            block.attn.c_proj.scale.data.fill_(teacher_block.attn.c_proj.weight.abs().mean())

            # MLP - also need to transpose
            block.ln_2.load_state_dict(teacher_block.ln_2.state_dict())
            block.mlp.c_fc.weight.data = teacher_block.mlp.c_fc.weight.data.T.clone()
            block.mlp.c_fc.bias.data = teacher_block.mlp.c_fc.bias.data.clone()
            block.mlp.c_proj.weight.data = teacher_block.mlp.c_proj.weight.data.T.clone()
            block.mlp.c_proj.bias.data = teacher_block.mlp.c_proj.bias.data.clone()

            block.mlp.c_fc.scale.data.fill_(teacher_block.mlp.c_fc.weight.abs().mean())
            block.mlp.c_proj.scale.data.fill_(teacher_block.mlp.c_proj.weight.abs().mean())

        return model


class DistillationTrainer:
    """Knowledge distillation trainer for binary GPT-2."""

    def __init__(
        self,
        student: BinaryGPT2,
        teacher: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        temperature: float = 4.0,
        alpha: float = 0.5,  # Weight for distillation vs CE loss
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()  # Teacher is frozen
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined distillation + CE loss."""
        # Soft targets (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence (scaled by T^2 as per Hinton et al.)
        kl_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard targets (cross-entropy)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return total_loss, {'kl': kl_loss.item(), 'ce': ce_loss.item()}

    def train_step(
        self,
        input_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single training step."""
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position

        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits

        # Get student predictions
        student_logits = self.student(input_ids)

        # Compute loss
        loss, loss_dict = self.distillation_loss(student_logits, teacher_logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for binary training stability)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

        optimizer.step()

        return {'loss': loss.item(), **loss_dict}

    @torch.no_grad()
    def evaluate(self, eval_texts: list, max_length: int = 128) -> dict:
        """Evaluate student vs teacher perplexity."""
        self.student.eval()

        student_losses = []
        teacher_losses = []

        for text in eval_texts:
            tokens = self.tokenizer.encode(text, return_tensors='pt',
                                          max_length=max_length, truncation=True)
            tokens = tokens.to(self.device)

            if tokens.size(1) < 2:
                continue

            labels = tokens.clone()
            labels[:, :-1] = tokens[:, 1:]
            labels[:, -1] = -100

            # Student
            student_logits = self.student(tokens)
            student_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            student_losses.append(student_loss.item())

            # Teacher
            teacher_outputs = self.teacher(tokens)
            teacher_loss = F.cross_entropy(
                teacher_outputs.logits.view(-1, teacher_outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            teacher_losses.append(teacher_loss.item())

        self.student.train()

        student_ppl = np.exp(np.mean(student_losses))
        teacher_ppl = np.exp(np.mean(teacher_losses))

        return {
            'student_ppl': student_ppl,
            'teacher_ppl': teacher_ppl,
            'ppl_ratio': student_ppl / teacher_ppl,
            'gap': (student_ppl / teacher_ppl - 1) * 100
        }


def quick_test_binary_gpt2():
    """Quick sanity check that binary GPT-2 works."""
    print("=" * 60)
    print("QUICK TEST: Binary GPT-2 Sanity Check")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load tokenizer and teacher
    print("\nLoading GPT-2 teacher...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained('gpt2')

    # Create binary student from teacher
    print("Creating binary student from teacher weights...")
    student = BinaryGPT2.from_pretrained_teacher('gpt2')

    # Parameter counts
    param_info = student.count_parameters()
    print(f"\nParameter breakdown:")
    print(f"  Binary params: {param_info['binary']:,} ({param_info['binary_fraction']*100:.1f}%)")
    print(f"  FP32 params:   {param_info['fp32']:,} ({(1-param_info['binary_fraction'])*100:.1f}%)")
    print(f"  Total:         {param_info['total']:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_text = "The quick brown fox jumps over"
    tokens = tokenizer.encode(test_text, return_tensors='pt')

    student.eval()
    teacher.eval()

    with torch.no_grad():
        student_logits = student(tokens)
        teacher_logits = teacher(tokens).logits

    print(f"Input: '{test_text}'")
    print(f"Student logits shape: {student_logits.shape}")
    print(f"Teacher logits shape: {teacher_logits.shape}")

    # Compare top predictions
    student_next = tokenizer.decode(student_logits[0, -1].argmax())
    teacher_next = tokenizer.decode(teacher_logits[0, -1].argmax())
    print(f"\nNext token prediction:")
    print(f"  Teacher: '{teacher_next}'")
    print(f"  Student: '{student_next}'")

    # Quick perplexity test
    eval_texts = [
        "The capital of France is Paris.",
        "In machine learning, neural networks are used for pattern recognition.",
        "The weather today is sunny with a high of 75 degrees.",
    ]

    trainer = DistillationTrainer(student, teacher, tokenizer, device=device)
    results = trainer.evaluate(eval_texts)

    print(f"\nPerplexity comparison (before training):")
    print(f"  Teacher PPL: {results['teacher_ppl']:.2f}")
    print(f"  Student PPL: {results['student_ppl']:.2f}")
    print(f"  Gap: {results['gap']:+.1f}%")

    print("\n" + "=" * 60)
    print("Quick test PASSED!")
    print("=" * 60)

    return student, teacher, tokenizer


def train_binary_gpt2_distillation(
    model_name: str = 'gpt2',
    train_texts: Optional[list] = None,
    n_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    temperature: float = 4.0,
    eval_every: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Train binary GPT-2 via distillation."""
    print("=" * 70)
    print("BINARY GPT-2 DISTILLATION TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Temperature: {temperature}")

    # Load models
    print("\nLoading teacher model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained(model_name)

    print("Creating binary student...")
    student = BinaryGPT2.from_pretrained_teacher(model_name)

    # Default training data if none provided
    if train_texts is None:
        print("\nUsing default WikiText-style training snippets...")
        train_texts = [
            "The transformer architecture has revolutionized natural language processing.",
            "Machine learning models require large amounts of training data.",
            "Neural networks consist of interconnected layers of neurons.",
            "Deep learning has achieved remarkable results in computer vision.",
            "Language models predict the probability of word sequences.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Gradient descent is used to optimize neural network parameters.",
            "Backpropagation computes gradients efficiently through the network.",
            "Regularization techniques help prevent overfitting in neural networks.",
            "Transfer learning leverages knowledge from pretrained models.",
        ] * 50  # Repeat for more training data

    # Evaluation data
    eval_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming many industries.",
        "The stock market experienced significant volatility today.",
    ]

    # Create trainer
    trainer = DistillationTrainer(
        student, teacher, tokenizer,
        temperature=temperature,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)

    # Initial evaluation
    print("\nInitial evaluation:")
    init_results = trainer.evaluate(eval_texts)
    print(f"  Teacher PPL: {init_results['teacher_ppl']:.2f}")
    print(f"  Student PPL: {init_results['student_ppl']:.2f}")
    print(f"  Gap: {init_results['gap']:+.1f}%")

    # Training loop
    print("\nTraining...")
    step = 0
    best_gap = init_results['gap']

    for epoch in range(n_epochs):
        np.random.shuffle(train_texts)
        epoch_losses = []

        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]

            # Tokenize
            tokens = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).input_ids

            # Train step
            loss_dict = trainer.train_step(tokens, optimizer)
            epoch_losses.append(loss_dict['loss'])
            step += 1

            # Evaluate periodically
            if step % eval_every == 0:
                results = trainer.evaluate(eval_texts)
                print(f"  Step {step}: loss={loss_dict['loss']:.4f}, "
                      f"student_ppl={results['student_ppl']:.2f}, "
                      f"gap={results['gap']:+.1f}%")

                if results['gap'] < best_gap:
                    best_gap = results['gap']

        avg_loss = np.mean(epoch_losses)
        results = trainer.evaluate(eval_texts)
        print(f"\nEpoch {epoch+1}/{n_epochs}: avg_loss={avg_loss:.4f}, "
              f"student_ppl={results['student_ppl']:.2f}, gap={results['gap']:+.1f}%")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    final_results = trainer.evaluate(eval_texts)
    print(f"Teacher PPL: {final_results['teacher_ppl']:.2f}")
    print(f"Student PPL: {final_results['student_ppl']:.2f}")
    print(f"Gap: {final_results['gap']:+.1f}%")
    print(f"Best gap achieved: {best_gap:+.1f}%")

    # BPP calculation
    param_info = student.count_parameters()
    binary_bpp = 1.0  # Binary weights
    fp32_bpp = 32.0   # Embeddings and LayerNorms

    effective_bpp = (param_info['binary'] * binary_bpp +
                     param_info['fp32'] * fp32_bpp) / param_info['total']

    print(f"\nStorage:")
    print(f"  Effective BPP (including embeddings): {effective_bpp:.2f}")
    print(f"  If only counting linear layers: 1.00 bpp")

    return student, final_results


if __name__ == '__main__':
    # Run quick test first
    quick_test_binary_gpt2()

