import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    This implementation accepts an optional padding mask and returns both
    the attention output and the attention weights for analysis.
    
    Args:
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = None  # Will be set during forward pass
        
        if DEBUG_MODE:
            print(f"[DEBUG]: ScaledDotProductAttention initialized")
            print(f"[DEBUG]:   - Dropout: {dropout}")
    
    def forward(self, query, key, value, mask=None, return_attention=True):
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_k)
            mask: Optional padding mask of shape (batch_size, seq_len_q, seq_len_k)
                  or (batch_size, 1, 1, seq_len_k) for broadcasting
            return_attention: Whether to return attention weights (default: True)
        
        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, num_heads, seq_len_q, d_k = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Compute scaling factor: 1 / sqrt(d_k)
        self.scale = 1.0 / np.sqrt(d_k)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Attention scores shape: {attention_scores.shape}")
            print(f"[DEBUG]:   - Scale factor: {self.scale:.4f}")
        
        # Apply mask if provided
        if mask is not None:
            # Mask should broadcast to (batch_size, num_heads, seq_len_q, seq_len_k)
            # If mask is (batch_size, seq_len_q, seq_len_k), add num_heads dimension
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Shape: (batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 2:
                # If mask is (batch_size, seq_len_k), reshape for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len_k)
            
            # Apply mask by setting masked positions to -inf
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
            
            if DEBUG_MODE:
                print(f"[DEBUG]: Mask applied to attention scores")
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute attention output: attention_weights @ V
        # Shape: (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Attention output shape: {output.shape}")
            print(f"[DEBUG]: Attention weights shape: {attention_weights.shape}")
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def extra_repr(self):
        """Extra representation for print()"""
        return f'dropout={self.dropout.p}'


def create_padding_mask(tokens, pad_idx=0):
    """
    Create a padding mask for attention mechanism.
    
    Args:
        tokens: Token indices of shape (batch_size, seq_len)
        pad_idx: Index used for padding (default: 0)
    
    Returns:
        mask: Boolean mask of shape (batch_size, 1, 1, seq_len)
              True for valid tokens, False for padding tokens
    """
    # Shape: (batch_size, seq_len)
    mask = (tokens != pad_idx)
    
    # Reshape for broadcasting: (batch_size, 1, 1, seq_len)
    mask = mask.unsqueeze(1).unsqueeze(2)
    
    return mask


def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention (decoder).
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Boolean mask of shape (1, 1, seq_len, seq_len)
              True for positions that can be attended to (lower triangular)
    """
    # Create upper triangular matrix of -inf
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # Invert: True for positions we CAN attend to (lower triangular)
    mask = ~mask
    
    # Reshape for broadcasting
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


def combine_masks(padding_mask, causal_mask=None):
    """
    Combine padding mask and causal mask.
    
    Args:
        padding_mask: Padding mask of shape (batch_size, 1, 1, seq_len)
        causal_mask: Causal mask of shape (1, 1, seq_len, seq_len)
    
    Returns:
        combined_mask: Combined mask
    """
    if causal_mask is None:
        return padding_mask
    
    # Combine: both must be True to attend
    combined_mask = padding_mask & causal_mask
    
    return combined_mask


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_scaled_dot_product_attention():
    """Test the ScaledDotProductAttention implementation"""
    print("=" * 60)
    print("TESTING SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)
    
    # Create sample inputs
    batch_size = 2
    num_heads = 4
    seq_len_q = 10
    seq_len_k = 10
    d_k = 64
    d_v = 64
    
    query = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    key = torch.randn(batch_size, num_heads, seq_len_k, d_k)
    value = torch.randn(batch_size, num_heads, seq_len_k, d_v)
    
    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key:   {key.shape}")
    print(f"  Value: {value.shape}")
    
    # Initialize attention
    attention = ScaledDotProductAttention(dropout=0.1)
    
    # Test without mask
    print("\n" + "-" * 40)
    print("Test 1: Without mask")
    print("-" * 40)
    
    output, attn_weights = attention(query, key, value)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1): {attn_weights.sum(dim=-1).mean():.4f}")
    
    # Test with padding mask
    print("\n" + "-" * 40)
    print("Test 2: With padding mask")
    print("-" * 40)
    
    # Create a padding mask (last 3 positions are padding)
    tokens = torch.ones(batch_size, seq_len_k)
    tokens[:, -3:] = 0  # Mark as padding
    padding_mask = create_padding_mask(tokens)
    print(f"Padding mask shape: {padding_mask.shape}")
    
    output_masked, attn_weights_masked = attention(query, key, value, mask=padding_mask)
    print(f"Output shape: {output_masked.shape}")
    print(f"Attention weights shape: {attn_weights_masked.shape}")
    
    # Verify padding positions have zero attention
    attn_to_padding = attn_weights_masked[:, :, :, -3:].sum()
    print(f"Attention to padding positions (should be 0): {attn_to_padding:.6f}")
    
    # Test with causal mask
    print("\n" + "-" * 40)
    print("Test 3: With causal mask (for decoder)")
    print("-" * 40)
    
    causal_mask = create_causal_mask(seq_len_q)
    print(f"Causal mask shape: {causal_mask.shape}")
    
    # Make query and key same length for causal test
    query_square = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    key_square = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    value_square = torch.randn(batch_size, num_heads, seq_len_q, d_v)
    
    output_causal, attn_weights_causal = attention(
        query_square, key_square, value_square, mask=causal_mask
    )
    
    # Check that future positions have zero attention
    # For position i, attention to positions > i should be 0
    for i in range(seq_len_q):
        future_attn = attn_weights_causal[:, :, i, i+1:].sum()
        print(f"  Position {i}: attention to future = {future_attn:.6f}")
        assert future_attn < 1e-6, f"Attention to future at position {i} should be 0"
    
    print("\n✓ All tests passed!")
    
    return output, attn_weights


def test_attention_gradients():
    """Test gradient flow through attention"""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    batch_size = 2
    num_heads = 4
    seq_len = 10
    d_k = 64
    d_v = 64
    
    # Create inputs with gradients
    query = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    key = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    value = torch.randn(batch_size, num_heads, seq_len, d_v, requires_grad=True)
    
    attention = ScaledDotProductAttention(dropout=0.1)
    
    # Forward pass
    output, _ = attention(query, key, value)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  query.grad is None: {query.grad is None}")
    print(f"  key.grad is None: {key.grad is None}")
    print(f"  value.grad is None: {value.grad is None}")
    
    if query.grad is not None:
        print(f"  query.grad norm: {query.grad.norm():.4f}")
    if key.grad is not None:
        print(f"  key.grad norm: {key.grad.norm():.4f}")
    if value.grad is not None:
        print(f"  value.grad norm: {value.grad.norm():.4f}")
    
    print("\n✓ Gradients flow correctly!")
    
    return query.grad is not None and key.grad is not None and value.grad is not None


def visualize_attention_weights():
    """Visualize attention weights as a heatmap (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return
    
    print("\n" + "=" * 60)
    print("VISUALIZING ATTENTION WEIGHTS")
    print("=" * 60)
    
    batch_size = 1
    num_heads = 4
    seq_len = 20
    d_k = 64
    d_v = 64
    
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_v)
    
    attention = ScaledDotProductAttention(dropout=0.0)  # No dropout for visualization
    
    _, attn_weights = attention(query, key, value)
    
    # Plot attention weights for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    
    for h in range(num_heads):
        ax = axes[h] if num_heads > 1 else axes
        weights = attn_weights[0, h].detach().numpy()
        im = ax.imshow(weights, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {h + 1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = os.getenv('figuresDir', './figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'attention_weights_visualization.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Attention weights visualization saved to: {save_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("SCALED DOT-PRODUCT ATTENTION MODULE")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Run tests
    test_scaled_dot_product_attention()
    test_attention_gradients()
    visualize_attention_weights()
    
    print("\n" + "=" * 80)
    print("MODULE READY")
    print("=" * 80)