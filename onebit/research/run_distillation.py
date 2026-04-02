"""Run binary GPT-2 distillation training."""
from binary_distillation import train_binary_gpt2_distillation

if __name__ == '__main__':
    # Run training with small dataset for quick test
    train_binary_gpt2_distillation(
        model_name='gpt2',
        n_epochs=5,
        batch_size=2,
        learning_rate=1e-4,
        temperature=4.0,
        eval_every=50,
    )

