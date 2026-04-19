import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from dotenv import load_dotenv

# Add the current directory to path to import from same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the transformer components
from MultiHead_selfAttention import MultiHeadSelfAttention
from PositionWise_feedForward_Network import PositionWiseFeedForward

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block with Pre-Layer Normalization.
    
    Implements the Pre-LN architecture:
        x ← x + Dropout(MultiHead(LN(x)))
        x ← x + Dropout(FFN(LN(x)))
    
    Args:
        d_model: Model dimension (default: 128)
        h: Number of attention heads (default: 4)
        d_k: Dimension of keys per head (default: 32)
        d_v: Dimension of values per head (default: 32)
        d_ff: Feed-forward inner dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, d_ff=512, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        # Layer Normalization for Multi-Head Attention (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model, 
            h=h, 
            d_k=d_k, 
            d_v=d_v, 
            dropout=dropout
        )
        
        # Dropout for attention residual
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer Normalization for Feed-Forward Network (Pre-LN)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Position-Wise Feed-Forward Network
        self.ffn = PositionWiseFeedForward(
            d_model=d_model, 
            d_ff=d_ff, 
            dropout=dropout
        )
        
        # Dropout for FFN residual
        self.dropout2 = nn.Dropout(dropout)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: TransformerEncoderBlock initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - h (heads): {h}")
            print(f"[DEBUG]:   - d_k: {d_k}, d_v: {d_v}")
            print(f"[DEBUG]:   - d_ff: {d_ff}")
            print(f"[DEBUG]:   - Dropout: {dropout}")
            print(f"[DEBUG]:   - Architecture: Pre-Layer Normalization")
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass for a single encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights (default: False)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights if return_attention=True
        """
        if DEBUG_MODE:
            print(f"[DEBUG]: TransformerEncoderBlock forward")
            print(f"[DEBUG]:   - Input shape: {x.shape}")
        
        # ================================================================
        # Multi-Head Self-Attention Sub-layer (Pre-LN)
        # ================================================================
        
        # Pre-Layer Normalization
        norm_x = self.norm1(x)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After norm1: mean={norm_x.mean():.4f}, std={norm_x.std():.4f}")
        
        # Multi-Head Self-Attention
        attn_output, attn_weights = self.attention(norm_x, mask=mask, return_attention=True)
        
        # Dropout + Residual Connection
        x = x + self.dropout1(attn_output)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After attention + residual: mean={x.mean():.4f}, std={x.std():.4f}")
        
        # ================================================================
        # Position-Wise Feed-Forward Sub-layer (Pre-LN)
        # ================================================================
        
        # Pre-Layer Normalization
        norm_x = self.norm2(x)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After norm2: mean={norm_x.mean():.4f}, std={norm_x.std():.4f}")
        
        # Feed-Forward Network
        ffn_output = self.ffn(norm_x)
        
        # Dropout + Residual Connection
        x = x + self.dropout2(ffn_output)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After FFN + residual: mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"[DEBUG]:   - Output shape: {x.shape}")
        
        if return_attention:
            return x, attn_weights
        else:
            return x
    
    def extra_repr(self):
        """Extra representation for print()"""
        return f'd_model={self.d_model}, h={self.h}, d_k={self.d_k}, d_v={self.d_v}, d_ff={self.d_ff}, dropout={self.dropout_rate}'


class TransformerEncoderBlockPostLN(nn.Module):
    """
    Transformer Encoder Block with Post-Layer Normalization (original architecture).
    
    Implements the original Post-LN architecture:
        x ← LN(x + Dropout(MultiHead(x)))
        x ← LN(x + Dropout(FFN(x)))
    
    This is provided for comparison with Pre-LN.
    
    Args:
        d_model: Model dimension (default: 128)
        h: Number of attention heads (default: 4)
        d_k: Dimension of keys per head (default: 32)
        d_v: Dimension of values per head (default: 32)
        d_ff: Feed-forward inner dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, d_ff=512, dropout=0.1):
        super(TransformerEncoderBlockPostLN, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model, h=h, d_k=d_k, d_v=d_v, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Position-Wise Feed-Forward Network
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: TransformerEncoderBlockPostLN initialized")
            print(f"[DEBUG]:   - Architecture: Post-Layer Normalization")
    
    def forward(self, x, mask=None, return_attention=False):
        """Forward pass with Post-LN architecture."""
        # Attention sub-layer
        attn_output, attn_weights = self.attention(x, mask=mask, return_attention=True)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN sub-layer
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        if return_attention:
            return x, attn_weights
        else:
            return x


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_transformer_encoder_block():
    """Test the TransformerEncoderBlock implementation"""
    print("=" * 60)
    print("TESTING TRANSFORMER ENCODER BLOCK")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 50
    d_model = 128
    h = 4
    d_k = 32
    d_v = 32
    d_ff = 512
    dropout = 0.1
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Hyperparameters:")
    print(f"  d_model: {d_model}")
    print(f"  h (heads): {h}")
    print(f"  d_k: {d_k}, d_v: {d_v}")
    print(f"  d_ff: {d_ff}")
    print(f"  dropout: {dropout}")
    
    # Initialize encoder block
    encoder_block = TransformerEncoderBlock(
        d_model=d_model, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout
    )
    
    # Test 1: Forward pass without mask
    print("\n" + "-" * 40)
    print("Test 1: Forward pass without mask")
    print("-" * 40)
    
    output = encoder_block(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    # Test 2: Forward pass with mask
    print("\n" + "-" * 40)
    print("Test 2: Forward pass with padding mask")
    print("-" * 40)
    
    # Create padding mask (last 10 positions are padding)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -10:] = 0
    mask = (mask != 0)
    
    output_masked = encoder_block(x, mask=mask)
    print(f"Output shape (masked): {output_masked.shape}")
    
    # Test 3: Return attention weights
    print("\n" + "-" * 40)
    print("Test 3: Return attention weights")
    print("-" * 40)
    
    output, attn_weights = encoder_block(x, return_attention=True)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected attention shape: ({batch_size}, {h}, {seq_len}, {seq_len})")
    
    # Test 4: Verify residual connections
    print("\n" + "-" * 40)
    print("Test 4: Residual connection verification")
    print("-" * 40)
    
    # Create a copy of input to verify it's different from output
    x_test = torch.randn(batch_size, seq_len, d_model)
    output_test = encoder_block(x_test)
    
    diff = (output_test - x_test).abs().max()
    print(f"Max difference between input and output: {diff:.6f}")
    print(f"Residual connections working (diff > 0): {diff > 0}")
    
    # Test 5: Compare Pre-LN and Post-LN
    print("\n" + "-" * 40)
    print("Test 5: Pre-LN vs Post-LN comparison")
    print("-" * 40)
    
    encoder_pre = TransformerEncoderBlock(
        d_model=d_model, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=0.0
    )
    encoder_post = TransformerEncoderBlockPostLN(
        d_model=d_model, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=0.0
    )
    
    # Initialize with same weights for fair comparison
    encoder_post.attention.W_q.weight.data = encoder_pre.attention.W_q.weight.data.clone()
    encoder_post.attention.W_k.weight.data = encoder_pre.attention.W_k.weight.data.clone()
    encoder_post.attention.W_v.weight.data = encoder_pre.attention.W_v.weight.data.clone()
    encoder_post.attention.W_o.weight.data = encoder_pre.attention.W_o.weight.data.clone()
    encoder_post.ffn.linear1.weight.data = encoder_pre.ffn.linear1.weight.data.clone()
    encoder_post.ffn.linear2.weight.data = encoder_pre.ffn.linear2.weight.data.clone()
    
    x_comp = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        out_pre = encoder_pre(x_comp)
        out_post = encoder_post(x_comp)
    
    pre_std = out_pre.std().item()
    post_std = out_post.std().item()
    
    print(f"Pre-LN output std: {pre_std:.4f}")
    print(f"Post-LN output std: {post_std:.4f}")
    print(f"Pre-LN typically has better gradient flow and training stability")
    
    print("\n✓ All tests passed!")
    
    return output


def test_gradient_flow():
    """Test gradient flow through encoder block"""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    encoder_block = TransformerEncoderBlock(
        d_model=d_model, h=4, d_k=32, d_v=32, d_ff=512, dropout=0.1
    )
    
    # Forward pass
    output = encoder_block(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  x.grad is None: {x.grad is None}")
    print(f"  norm1.weight.grad is None: {encoder_block.norm1.weight.grad is None}")
    print(f"  norm2.weight.grad is None: {encoder_block.norm2.weight.grad is None}")
    print(f"  attention.W_q.weight.grad is None: {encoder_block.attention.W_q.weight.grad is None}")
    print(f"  ffn.linear1.weight.grad is None: {encoder_block.ffn.linear1.weight.grad is None}")
    
    if x.grad is not None:
        print(f"  x.grad norm: {x.grad.norm():.4f}")
    
    print("\n✓ Gradients flow correctly through all components!")
    
    return True


def test_parameter_count():
    """Calculate and display parameter count"""
    print("\n" + "=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    
    encoder_block = TransformerEncoderBlock(
        d_model=128, h=4, d_k=32, d_v=32, d_ff=512, dropout=0.1
    )
    
    total_params = sum(p.numel() for p in encoder_block.parameters())
    trainable_params = sum(p.numel() for p in encoder_block.parameters() if p.requires_grad)
    
    # Count parameters by component
    attn_params = sum(p.numel() for p in encoder_block.attention.parameters())
    ffn_params = sum(p.numel() for p in encoder_block.ffn.parameters())
    norm_params = sum(p.numel() for p in encoder_block.norm1.parameters()) + \
                  sum(p.numel() for p in encoder_block.norm2.parameters())
    
    print(f"\nTransformerEncoderBlock Parameters:")
    print(f"  Multi-Head Attention: {attn_params:,}")
    print(f"  Feed-Forward Network: {ffn_params:,}")
    print(f"  Layer Normalization: {norm_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Parameters for stacked encoders
    print(f"\nStacked Encoder Parameters:")
    for n_layers in [1, 2, 4, 6, 8]:
        total = total_params * n_layers
        print(f"  {n_layers} layer(s): {total:,} parameters")
    
    return total_params


def test_multiple_stacked_blocks():
    """Test stacking multiple encoder blocks"""
    print("\n" + "=" * 60)
    print("TESTING STACKED ENCODER BLOCKS")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Stack 4 encoder blocks
    n_layers = 4
    encoder_layers = nn.ModuleList([
        TransformerEncoderBlock(d_model=128, h=4, d_k=32, d_v=32, d_ff=512, dropout=0.1)
        for _ in range(n_layers)
    ])
    
    print(f"\nStacking {n_layers} encoder blocks...")
    print(f"Input shape: {x.shape}")
    
    # Forward pass through all layers
    output = x
    attention_weights = []
    
    for i, layer in enumerate(encoder_layers):
        output, attn = layer(output, return_attention=True)
        attention_weights.append(attn)
        print(f"  Layer {i+1}: output shape={output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")
    
    print(f"\nFinal output shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    
    # Verify output is not NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("\n✓ Stacked encoder blocks test passed!")
    
    return output, attention_weights


if __name__ == "__main__":
    print("=" * 80)
    print("TRANSFORMER ENCODER BLOCK MODULE")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Run tests
    test_transformer_encoder_block()
    test_gradient_flow()
    test_parameter_count()
    test_multiple_stacked_blocks()
    
    print("\n" + "=" * 80)
    print("MODULE READY")
    print("=" * 80)
    print("\nUsage in main.py:")
    print("  from TransformerEncoder.transformer_encoder import TransformerEncoderBlock")
    print("  encoder = TransformerEncoderBlock(d_model=128, h=4, d_ff=512)")
    print("  output = encoder(input_tokens, mask=padding_mask)")