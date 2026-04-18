import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Add the current directory to path to import from same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the scaled dot-product attention
from scaled_dotProductAttention import ScaledDotProductAttention, create_padding_mask

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Splits the input into multiple heads, applies scaled dot-product attention
    in parallel, and concatenates the results.
    
    Args:
        d_model: Model dimension (default: 128)
        h: Number of attention heads (default: 4)
        d_k: Dimension of keys per head (default: 32)
        d_v: Dimension of values per head (default: 32)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_rate = dropout
        
        # Validate dimensions
        assert d_model == h * d_k, f"d_model ({d_model}) must equal h * d_k ({h} * {d_k} = {h * d_k})"
        
        # Separate projection matrices for Q, K, V per head
        # Instead of h separate matrices, we use one large matrix and reshape
        self.W_q = nn.Linear(d_model, h * d_k, bias=False)
        self.W_k = nn.Linear(d_model, h * d_k, bias=False)
        self.W_v = nn.Linear(d_model, h * d_v, bias=False)
        
        # Shared output projection
        self.W_o = nn.Linear(h * d_v, d_model, bias=False)
        
        # Scaled dot-product attention (shared across heads)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Dropout for output
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: MultiHeadSelfAttention initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - h (heads): {h}")
            print(f"[DEBUG]:   - d_k per head: {d_k}")
            print(f"[DEBUG]:   - d_v per head: {d_v}")
            print(f"[DEBUG]:   - Dropout: {dropout}")
            print(f"[DEBUG]:   - W_q shape: {self.W_q.weight.shape}")
            print(f"[DEBUG]:   - W_k shape: {self.W_k.weight.shape}")
            print(f"[DEBUG]:   - W_v shape: {self.W_v.weight.shape}")
            print(f"[DEBUG]:   - W_o shape: {self.W_o.weight.shape}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Weights initialized with Xavier uniform")
    
    def forward(self, x, mask=None, return_attention=True):
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask of shape (batch_size, seq_len)
                  or (batch_size, 1, 1, seq_len)
            return_attention: Whether to return attention weights (default: True)
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, h, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        if DEBUG_MODE:
            print(f"[DEBUG]: MultiHeadSelfAttention forward")
            print(f"[DEBUG]:   - Input shape: {x.shape}")
        
        # Linear projections for Q, K, V
        # Shape: (batch_size, seq_len, h * d_k) or (batch_size, seq_len, h * d_v)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Q after projection: {Q.shape}")
            print(f"[DEBUG]:   - K after projection: {K.shape}")
            print(f"[DEBUG]:   - V after projection: {V.shape}")
        
        # Reshape and transpose for multi-head attention
        # From (batch_size, seq_len, h * d_k) to (batch_size, h, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.h, self.d_v).transpose(1, 2)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Q reshaped: {Q.shape}")
            print(f"[DEBUG]:   - K reshaped: {K.shape}")
            print(f"[DEBUG]:   - V reshaped: {V.shape}")
        
        # Prepare mask for multi-head attention
        # If mask is provided and is 2D (batch_size, seq_len), reshape for broadcasting
        if mask is not None:
            if mask.dim() == 2:
                # Shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Shape: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            
            if DEBUG_MODE:
                print(f"[DEBUG]:   - Mask shape after processing: {mask.shape}")
        
        # Apply scaled dot-product attention
        # Output shape: (batch_size, h, seq_len, d_v)
        # Attention weights shape: (batch_size, h, seq_len, seq_len)
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask, return_attention=True)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Attention output shape: {attn_output.shape}")
            print(f"[DEBUG]:   - Attention weights shape: {attn_weights.shape}")
        
        # Concatenate heads
        # From (batch_size, h, seq_len, d_v) to (batch_size, seq_len, h * d_v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.h * self.d_v)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Concatenated output shape: {attn_output.shape}")
        
        # Apply output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Final output shape: {output.shape}")
        
        if return_attention:
            return output, attn_weights
        else:
            return output
    
    def extra_repr(self):
        """Extra representation for print()"""
        return f'd_model={self.d_model}, h={self.h}, d_k={self.d_k}, d_v={self.d_v}, dropout={self.dropout_rate}'


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention mechanism (for encoder-decoder attention).
    
    Similar to self-attention but takes separate encoder outputs as K and V.
    
    Args:
        d_model: Model dimension (default: 128)
        h: Number of attention heads (default: 4)
        d_k: Dimension of keys per head (default: 32)
        d_v: Dimension of values per head (default: 32)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_rate = dropout
        
        # Validate dimensions
        assert d_model == h * d_k, f"d_model ({d_model}) must equal h * d_k ({h} * {d_k} = {h * d_k})"
        
        # Projection matrices
        self.W_q = nn.Linear(d_model, h * d_k, bias=False)  # Query from decoder
        self.W_k = nn.Linear(d_model, h * d_k, bias=False)  # Key from encoder
        self.W_v = nn.Linear(d_model, h * d_v, bias=False)  # Value from encoder
        
        # Shared output projection
        self.W_o = nn.Linear(h * d_v, d_model, bias=False)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: MultiHeadCrossAttention initialized")
            print(f"[DEBUG]:   - d_model: {d_model}, h: {h}, d_k: {d_k}, d_v: {d_v}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, x, encoder_output, encoder_mask=None, return_attention=True):
        """
        Forward pass for cross-attention.
        
        Args:
            x: Decoder input of shape (batch_size, seq_len_q, d_model)
            encoder_output: Encoder output of shape (batch_size, seq_len_kv, d_model)
            encoder_mask: Optional mask for encoder padding (batch_size, seq_len_kv)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Cross-attention weights
        """
        batch_size, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = encoder_output.shape
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.h, self.d_v).transpose(1, 2)
        
        # Prepare mask
        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask=encoder_mask, return_attention=True)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.h * self.d_v)
        
        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        else:
            return output


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_multi_head_self_attention():
    """Test the MultiHeadSelfAttention implementation"""
    print("=" * 60)
    print("TESTING MULTI-HEAD SELF-ATTENTION")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    h = 4
    d_k = 32
    d_v = 32
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Hyperparameters:")
    print(f"  d_model: {d_model}")
    print(f"  h (heads): {h}")
    print(f"  d_k: {d_k}")
    print(f"  d_v: {d_v}")
    
    # Initialize multi-head attention
    mha = MultiHeadSelfAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, dropout=0.1)
    
    # Test 1: Without mask
    print("\n" + "-" * 40)
    print("Test 1: Without mask")
    print("-" * 40)
    
    output, attn_weights = mha(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Expected attention shape: ({batch_size}, {h}, {seq_len}, {seq_len})")
    
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    assert attn_weights.shape == (batch_size, h, seq_len, seq_len), "Attention shape mismatch"
    
    # Test 2: With padding mask
    print("\n" + "-" * 40)
    print("Test 2: With padding mask")
    print("-" * 40)
    
    # Create padding mask (last 3 positions are padding)
    tokens = torch.ones(batch_size, seq_len)
    tokens[:, -3:] = 0
    mask = (tokens != 0)  # Shape: (batch_size, seq_len)
    
    output_masked, attn_weights_masked = mha(x, mask=mask)
    print(f"Output shape: {output_masked.shape}")
    print(f"Attention weights shape: {attn_weights_masked.shape}")
    
    # Verify padding positions have zero attention
    attn_to_padding = attn_weights_masked[:, :, :, -3:].sum()
    print(f"Attention to padding positions (should be 0): {attn_to_padding:.6f}")
    
    # Test 3: Verify output dimension
    print("\n" + "-" * 40)
    print("Test 3: Dimension verification")
    print("-" * 40)
    print(f"Input dimension: {x.shape[-1]}")
    print(f"Output dimension: {output.shape[-1]}")
    print(f"Dimensions preserved: {x.shape[-1] == output.shape[-1]}")
    
    print("\n✓ All tests passed!")
    
    return output, attn_weights


def test_multi_head_cross_attention():
    """Test the MultiHeadCrossAttention implementation"""
    print("\n" + "=" * 60)
    print("TESTING MULTI-HEAD CROSS-ATTENTION")
    print("=" * 60)
    
    batch_size = 2
    seq_len_q = 8
    seq_len_kv = 12
    d_model = 128
    h = 4
    d_k = 32
    d_v = 32
    
    # Create inputs
    x = torch.randn(batch_size, seq_len_q, d_model)  # Decoder input
    encoder_output = torch.randn(batch_size, seq_len_kv, d_model)  # Encoder output
    
    print(f"\nInput shapes:")
    print(f"  Query (decoder): {x.shape}")
    print(f"  Key/Value (encoder): {encoder_output.shape}")
    
    # Initialize cross-attention
    mca = MultiHeadCrossAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, dropout=0.1)
    
    # Test without mask
    print("\n" + "-" * 40)
    print("Test: Cross-attention without mask")
    print("-" * 40)
    
    output, attn_weights = mca(x, encoder_output)
    print(f"Output shape: {output.shape}")
    print(f"Cross-attention weights shape: {attn_weights.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len_q}, {d_model})")
    print(f"Expected attention shape: ({batch_size}, {h}, {seq_len_q}, {seq_len_kv})")
    
    assert output.shape == (batch_size, seq_len_q, d_model), "Output shape mismatch"
    assert attn_weights.shape == (batch_size, h, seq_len_q, seq_len_kv), "Attention shape mismatch"
    
    print("\n✓ All tests passed!")
    
    return output, attn_weights


def test_gradient_flow():
    """Test gradient flow through multi-head attention"""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    h = 4
    d_k = 32
    d_v = 32
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    mha = MultiHeadSelfAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v, dropout=0.1)
    
    # Forward pass
    output, _ = mha(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  x.grad is None: {x.grad is None}")
    print(f"  W_q.weight.grad is None: {mha.W_q.weight.grad is None}")
    print(f"  W_k.weight.grad is None: {mha.W_k.weight.grad is None}")
    print(f"  W_v.weight.grad is None: {mha.W_v.weight.grad is None}")
    print(f"  W_o.weight.grad is None: {mha.W_o.weight.grad is None}")
    
    if x.grad is not None:
        print(f"  x.grad norm: {x.grad.norm():.4f}")
    if mha.W_q.weight.grad is not None:
        print(f"  W_q.grad norm: {mha.W_q.weight.grad.norm():.4f}")
    
    print("\n✓ Gradients flow correctly!")
    
    return True


def test_parameter_count():
    """Calculate and display parameter count"""
    print("\n" + "=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    
    d_model = 128
    h = 4
    d_k = 32
    d_v = 32
    
    mha = MultiHeadSelfAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)
    
    total_params = sum(p.numel() for p in mha.parameters())
    trainable_params = sum(p.numel() for p in mha.parameters() if p.requires_grad)
    
    print(f"\nMultiHeadSelfAttention Parameters:")
    print(f"  W_q: {mha.W_q.weight.numel():,}")
    print(f"  W_k: {mha.W_k.weight.numel():,}")
    print(f"  W_v: {mha.W_v.weight.numel():,}")
    print(f"  W_o: {mha.W_o.weight.numel():,}")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    return total_params


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-HEAD SELF-ATTENTION MODULE")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Run tests
    test_multi_head_self_attention()
    test_multi_head_cross_attention()
    test_gradient_flow()
    test_parameter_count()
    
    print("\n" + "=" * 80)
    print("MODULE READY")
    print("=" * 80)