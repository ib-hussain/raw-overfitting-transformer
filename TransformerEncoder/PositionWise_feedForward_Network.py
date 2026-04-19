import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applies two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Input/output dimension (default: 128)
        d_ff: Inner feed-forward dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        # First linear layer: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionWiseFeedForward initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - d_ff: {d_ff}")
            print(f"[DEBUG]:   - Dropout: {dropout}")
            print(f"[DEBUG]:   - Linear1: ({d_model}, {d_ff})")
            print(f"[DEBUG]:   - Linear2: ({d_ff}, {d_model})")
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        The original Transformer uses normal distribution with std=0.02,
        but Xavier uniform works well and is standard in PyTorch.
        """
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Weights initialized with Xavier uniform, biases zero")
    
    def forward(self, x):
        """
        Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionWiseFeedForward forward")
            print(f"[DEBUG]:   - Input shape: {x.shape}")
        
        # First linear transformation + activation
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After linear1 + ReLU: {hidden.shape}")
            print(f"[DEBUG]:   - Hidden stats: mean={hidden.mean():.4f}, std={hidden.std():.4f}")
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Second linear transformation
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        output = self.linear2(hidden)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - After linear2: {output.shape}")
            print(f"[DEBUG]:   - Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
        
        return output
    
    def extra_repr(self):
        """Extra representation for print()"""
        return f'd_model={self.d_model}, d_ff={self.d_ff}, dropout={self.dropout_rate}'


class PositionWiseFeedForwardGELU(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation.
    
    GELU is used in more recent Transformer variants (BERT, GPT).
    FFN(x) = GELU(xW1 + b1)W2 + b2
    
    Args:
        d_model: Input/output dimension (default: 128)
        d_ff: Inner feed-forward dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super(PositionWiseFeedForwardGELU, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionWiseFeedForwardGELU initialized")
            print(f"[DEBUG]:   - d_model: {d_model}, d_ff: {d_ff}, dropout: {dropout}")
    
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        """Forward pass with GELU activation"""
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        return output


class PositionWiseFeedForwardSwish(nn.Module):
    """
    Position-wise Feed-Forward Network with Swish/SiLU activation.
    
    Swish(x) = x * sigmoid(x)
    
    Args:
        d_model: Input/output dimension (default: 128)
        d_ff: Inner feed-forward dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super(PositionWiseFeedForwardSwish, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # SiLU is Swish
        
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionWiseFeedForwardSwish initialized")
            print(f"[DEBUG]:   - d_model: {d_model}, d_ff: {d_ff}")
    
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        """Forward pass with Swish activation"""
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        return output


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_position_wise_feed_forward():
    """Test the PositionWiseFeedForward implementation"""
    print("=" * 60)
    print("TESTING POSITION-WISE FEED-FORWARD NETWORK")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Hyperparameters:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    
    # Initialize FFN
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
    
    # Test forward pass
    print("\n" + "-" * 40)
    print("Test 1: Forward pass")
    print("-" * 40)
    
    output = ffn(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    
    # Test that the FFN is applied position-wise (independently per token)
    print("\n" + "-" * 40)
    print("Test 2: Position-wise independence")
    print("-" * 40)
    
    # Create input where first token is different, others same
    x_test = torch.zeros(batch_size, seq_len, d_model)
    x_test[:, 0, 0] = 1.0  # Modify first token
    x_test[:, 1:, 0] = 2.0  # Other tokens have different value
    
    ffn.eval()
    with torch.no_grad():
        output_test = ffn(x_test)
    
    # Check that tokens at positions >= 1 have identical outputs
    token1_output = output_test[:, 1, :]
    token2_output = output_test[:, 2, :]
    diff = (token1_output - token2_output).abs().max()
    
    print(f"Max difference between position 1 and 2: {diff:.6f}")
    print(f"Tokens processed independently: {diff < 1e-5}")
    
    # Test with different activation functions
    print("\n" + "-" * 40)
    print("Test 3: Different activation functions")
    print("-" * 40)
    
    ffn_gelu = PositionWiseFeedForwardGELU(d_model=d_model, d_ff=d_ff, dropout=0.1)
    ffn_swish = PositionWiseFeedForwardSwish(d_model=d_model, d_ff=d_ff, dropout=0.1)
    
    with torch.no_grad():
        out_relu = ffn(x)
        out_gelu = ffn_gelu(x)
        out_swish = ffn_swish(x)
    
    print(f"ReLU output: mean={out_relu.mean():.4f}, std={out_relu.std():.4f}")
    print(f"GELU output: mean={out_gelu.mean():.4f}, std={out_gelu.std():.4f}")
    print(f"Swish output: mean={out_swish.mean():.4f}, std={out_swish.std():.4f}")
    
    print("\n✓ All tests passed!")
    
    return output


def test_gradient_flow():
    """Test gradient flow through FFN"""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
    
    # Forward pass
    output = ffn(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  x.grad is None: {x.grad is None}")
    print(f"  linear1.weight.grad is None: {ffn.linear1.weight.grad is None}")
    print(f"  linear2.weight.grad is None: {ffn.linear2.weight.grad is None}")
    print(f"  linear1.bias.grad is None: {ffn.linear1.bias.grad is None}")
    print(f"  linear2.bias.grad is None: {ffn.linear2.bias.grad is None}")
    
    if x.grad is not None:
        print(f"  x.grad norm: {x.grad.norm():.4f}")
    if ffn.linear1.weight.grad is not None:
        print(f"  linear1.weight.grad norm: {ffn.linear1.weight.grad.norm():.4f}")
    if ffn.linear2.weight.grad is not None:
        print(f"  linear2.weight.grad norm: {ffn.linear2.weight.grad.norm():.4f}")
    
    print("\n✓ Gradients flow correctly!")
    
    return True


def test_parameter_count():
    """Calculate and display parameter count"""
    print("\n" + "=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    
    d_model = 128
    d_ff = 512
    
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
    
    total_params = sum(p.numel() for p in ffn.parameters())
    trainable_params = sum(p.numel() for p in ffn.parameters() if p.requires_grad)
    
    print(f"\nPositionWiseFeedForward Parameters:")
    print(f"  linear1.weight: {ffn.linear1.weight.numel():,} ({d_model} × {d_ff})")
    print(f"  linear1.bias: {ffn.linear1.bias.numel():,}")
    print(f"  linear2.weight: {ffn.linear2.weight.numel():,} ({d_ff} × {d_model})")
    print(f"  linear2.bias: {ffn.linear2.bias.numel():,}")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Compare with attention parameters
    attn_params = 65536  # From MultiHeadSelfAttention
    print(f"\nComparison:")
    print(f"  MultiHeadSelfAttention: {attn_params:,} parameters")
    print(f"  PositionWiseFeedForward: {total_params:,} parameters")
    print(f"  FFN is {total_params / attn_params:.1f}x larger than attention")
    
    return total_params


def benchmark_ffn():
    """Benchmark FFN forward pass speed"""
    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    
    import time
    
    batch_size = 32
    seq_len = 256
    d_model = 128
    d_ff = 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
    ffn.eval()
    
    # Warm-up
    for _ in range(10):
        _ = ffn(x)
    
    # Benchmark
    n_iterations = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = ffn(x)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nBenchmark results:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Time per iteration: {elapsed / n_iterations * 1000:.2f}ms")
    
    return elapsed


def test_different_d_ff_values():
    """Test FFN with different d_ff values"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT D_FF VALUES")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    d_ff_values = [128, 256, 512, 1024, 2048]
    
    print(f"\n{'d_ff':<10} {'Parameters':<15} {'Output Mean':<12} {'Output Std':<12}")
    print("-" * 50)
    
    for d_ff in d_ff_values:
        ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        params = sum(p.numel() for p in ffn.parameters())
        
        with torch.no_grad():
            output = ffn(x)
        
        print(f"{d_ff:<10} {params:<15,} {output.mean():<12.4f} {output.std():<12.4f}")
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    print("=" * 80)
    print("POSITION-WISE FEED-FORWARD NETWORK MODULE")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Run tests
    test_position_wise_feed_forward()
    test_gradient_flow()
    test_parameter_count()
    test_different_d_ff_values()
    benchmark_ffn()
    
    print("\n" + "=" * 80)
    print("MODULE READY")
    print("=" * 80)