"""
Extract attention patterns from transformer model.
"""

import warnings
from typing import Dict, Tuple

import torch
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")


class AttentionExtractor:
    """Extract and process attention patterns from GPT-2."""

    def __init__(self, model_name: str = "gpt2-small", device: str = None):
        """
        Initialize the attention extractor.

        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=self.device,
        )

        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        print(f"✓ Model loaded: {self.n_layers} layers, {self.n_heads} heads per layer")

    def get_attention_patterns(self, prompt: str) -> Tuple[torch.Tensor, list]:
        """
        Extract attention patterns for a given prompt.

        Args:
            prompt: Input text string

        Returns:
            attention_patterns: [n_layers, n_heads, seq_len, seq_len]
            token_strings: List of token strings
        """
        # Tokenize
        tokens = self.model.to_tokens(prompt)
        token_strings = self.model.to_str_tokens(prompt)

        # Run with cache
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)

        # Extract attention patterns (remove batch dimension)
        attention_patterns = cache.stack_activation("pattern")[
            0
        ]  # [layers, heads, seq, seq]

        return attention_patterns.cpu(), token_strings

    def extract_features(self, attention_patterns: torch.Tensor) -> Dict[str, float]:
        """
        Extract interpretable features from attention patterns.

        Args:
            attention_patterns: [n_layers, n_heads, seq_len, seq_len]

        Returns:
            Dictionary of features
        """
        features = {}
        n_layers, n_heads, seq_len, _ = attention_patterns.shape

        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attention_patterns[layer, head]
                prefix = f"L{layer}H{head}"

                # 1. Attention entropy (measure of diffusion/focus)
                # Higher entropy = more distributed attention
                attn_safe = attn + 1e-10  # Avoid log(0)
                entropy = -torch.sum(attn * torch.log(attn_safe), dim=-1).mean()
                features[f"entropy_{prefix}"] = entropy.item()

                # 2. Induction score (attending to previous token)
                if seq_len > 1:
                    # Diagonal offset -1 captures attention to previous position
                    induction = torch.diagonal(attn, offset=-1, dim1=0, dim2=1).mean()
                    features[f"induction_{prefix}"] = induction.item()
                else:
                    features[f"induction_{prefix}"] = 0.0

                # 3. Self-attention (diagonal)
                self_attn = torch.diagonal(attn, dim1=0, dim2=1).mean()
                features[f"self_attn_{prefix}"] = self_attn.item()

                # 4. First token attention (often special/BOS token)
                first_tok_attn = attn[:, 0].mean()
                features[f"first_tok_{prefix}"] = first_tok_attn.item()

                # 5. Attention spread (standard deviation)
                spread = attn.std()
                features[f"spread_{prefix}"] = spread.item()

                # 6. Max attention (how focused is the strongest attention)
                max_attn = attn.max(dim=-1)[0].mean()
                features[f"max_attn_{prefix}"] = max_attn.item()

        return features


if __name__ == "__main__":
    # Quick test
    extractor = AttentionExtractor()

    test_prompts = [
        "The capital of France is",
        "15 + 27 =",
        "If all dogs are mammals and Rex is a dog, then",
        "Once upon a time in a magical forest,",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        attn, tokens = extractor.get_attention_patterns(prompt)
        features = extractor.extract_features(attn)
        print(f"  Extracted {len(features)} features")
        print("  Sample features:")
        for i, (k, v) in enumerate(list(features.items())[:5]):
            print(f"    {k}: {v:.4f}")
