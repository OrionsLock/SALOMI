import torch
import numpy as np
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from onebit.model.quantize_gpt2 import load_quantized_model
from onebit.model.runtime_transformer import RuntimeTransformer, InferenceConfig
from pathlib import Path

def get_fp32_activations(model_name, input_ids):
    print("Running FP32 Teacher...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    activations = {}
    
    def get_hook(name):
        def hook(module, input, output):
            # output is usually a tuple (hidden_states, present_key_value_states)
            # or just hidden_states depending on layer
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            activations[name] = h.detach().numpy()
        return hook

    # Hook specific layers
    # h.0, h.5, h.11 (final)
    hooks = []
    for i in [0, 5, 11]:
        layer = model.transformer.h[i]
        hooks.append(layer.register_forward_hook(get_hook(f"h.{i}")))
    
    hooks.append(model.transformer.ln_f.register_forward_hook(get_hook("ln_f")))
    
    with torch.no_grad():
        _ = model(torch.tensor([input_ids]))
        
    for h in hooks:
        h.remove()
        
    return activations

def run_1bit_with_hooks(model_path, input_ids, T):
    print("Running 1-bit Student...")
    model = load_quantized_model(Path(model_path))
    
    # We need to hack RuntimeTransformer to expose intermediate states
    # or subclass it. Let's subclass for debugging.
    
    class DebugRuntime(RuntimeTransformer):
        def __init__(self, model, cfg):
            super().__init__(model, cfg)
            self.activations = {}
            
        def _attention_layer(self, x, layer_idx, seed):
            out = super()._attention_layer(x, layer_idx, seed)
            # We don't have easy access to the block output here since _attention_layer
            # returns x + attention_out.
            # But forward() calls _attention_layer then _ffn_layer.
            # We want the output of the BLOCK (after FFN).
            return out
            
        def forward(self, input_ids, seed=None, return_all_logits=False):
            if seed is None: seed = self.infer_cfg.seed
            seq_len = len(input_ids)
            
            # Embeddings
            wte = self.model.weights_fp32["wte"]
            wpe = self.model.weights_fp32["wpe"]
            x = wte[input_ids] + wpe[:seq_len]
            
            for layer_idx in range(self.cfg.n_layers):
                prefix = f"h.{layer_idx}"
                
                # Re-implement block logic to capture state
                ln1_g = self.model.weights_fp32[f"{prefix}.ln_1.g"]
                ln1_b = self.model.weights_fp32[f"{prefix}.ln_1.b"]
                x_norm = np.array([self._layer_norm(x[i], ln1_g, ln1_b) for i in range(seq_len)])
                
                x = self._attention_layer(x_norm, layer_idx, seed + layer_idx * 1000)
                
                ln2_g = self.model.weights_fp32[f"{prefix}.ln_2.g"]
                ln2_b = self.model.weights_fp32[f"{prefix}.ln_2.b"]
                x_norm = np.array([self._layer_norm(x[i], ln2_g, ln2_b) for i in range(seq_len)])
                
                x = self._ffn_layer(x_norm, layer_idx, seed + layer_idx * 1000 + 100)
                
                # Capture block output (only for interesting layers)
                if layer_idx in [0, 5, 11]:
                    self.activations[f"h.{layer_idx}"] = x.copy()
            
            ln_f_g = self.model.weights_fp32["ln_f.g"]
            ln_f_b = self.model.weights_fp32["ln_f.b"]
            x_final = np.array([self._layer_norm(x[i], ln_f_g, ln_f_b) for i in range(seq_len)])
            self.activations["ln_f"] = x_final
            
            return x_final # Return hidden state, not logits for this test

    infer_cfg = InferenceConfig(
        T=T, backend="cpu", order=2, beta=0.30, lambd=0.0, 
        walsh_N=2, antithetic=True, use_ctg=False, use_hcl_logits=True
    )
    
    runtime = DebugRuntime(model, infer_cfg)
    _ = runtime.forward(input_ids)
    
    return runtime.activations

def compare_activations(fp32_acts, onebit_acts):
    print("\n=== Layer-wise Comparison ===")
    for name in fp32_acts:
        if name not in onebit_acts:
            continue
            
        a_fp = fp32_acts[name][0] # Remove batch dim [1, Seq, D] -> [Seq, D]
        a_1b = onebit_acts[name]  # [Seq, D]
        
        # Compare last token
        t_fp = a_fp[-1]
        t_1b = a_1b[-1]
        
        corr = np.corrcoef(t_fp, t_1b)[0, 1]
        mag_ratio = np.linalg.norm(t_1b) / (np.linalg.norm(t_fp) + 1e-9)
        
        print(f"[{name}]")
        print(f"  Corr (Last Token): {corr:.4f}")
        print(f"  Mag Ratio:         {mag_ratio:.4f}")
        print(f"  FP32 Stats: mean={t_fp.mean():.4f}, std={t_fp.std():.4f}")
        print(f"  1Bit Stats: mean={t_1b.mean():.4f}, std={t_1b.std():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--T", type=int, default=32)
    args = parser.parse_args()
    
    prompt = "The capital of France is Paris."
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt)
    
    fp32_acts = get_fp32_activations("gpt2", input_ids)
    onebit_acts = run_1bit_with_hooks(args.model, input_ids, args.T)
    
    compare_activations(fp32_acts, onebit_acts)

if __name__ == "__main__":
    main()


