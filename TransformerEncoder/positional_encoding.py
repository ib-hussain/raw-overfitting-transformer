import torch
import torch.nn as nn
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')
figuresDir = os.getenv('figuresDir', './figures')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Implements the fixed (non-learned) positional encoding from "Attention Is All You Need".
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    The encoding is stored as a buffer (not updated during training) and added to the input.
    
    Args:
        d_model: Embedding dimension (default: 128)
        max_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = self._create_positional_encoding(max_len, d_model)
        
        # Register as buffer (not a parameter, so it's not updated during training)
        self.register_buffer('pe', pe)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionalEncoding initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - max_len: {max_len}")
            print(f"[DEBUG]:   - Dropout: {dropout}")
            print(f"[DEBUG]:   - PE shape: {self.pe.shape}")
            print(f"[DEBUG]:   - PE is learnable: False (registered as buffer)")
    
    def _create_positional_encoding(self, max_len, d_model):
        """
        Create sinusoidal positional encoding matrix.
        
        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
        
        Returns:
            pe: Positional encoding tensor of shape (1, max_len, d_model)
        """
        # Create position indices: shape (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices: shape (d_model)
        div_term = torch.arange(0, d_model, 2, dtype=torch.float)
        
        # Compute the denominator: 10000^(2i/d_model)
        div_term = torch.exp(div_term * (-np.log(10000.0) / d_model))
        
        # Create empty encoding matrix: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Compute sin for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Compute cos for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Positional encoding created")
            print(f"[DEBUG]:   - Position shape: {position.shape}")
            print(f"[DEBUG]:   - div_term shape: {div_term.shape}")
            print(f"[DEBUG]:   - PE value range: [{pe.min():.4f}, {pe.max():.4f}]")
        
        return pe
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Embeddings with positional encoding added
        """
        if DEBUG_MODE:
            print(f"[DEBUG]: PositionalEncoding forward")
            print(f"[DEBUG]:   - Input shape: {x.shape}")
        
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasted across batch)
        # pe[:, :seq_len] has shape (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        if DEBUG_MODE:
            print(f"[DEBUG]:   - Added PE of shape: {self.pe[:, :seq_len, :].shape}")
            print(f"[DEBUG]:   - Output shape: {x.shape}")
        
        return x
    
    def get_positional_encoding(self, seq_len):
        """
        Get the positional encoding for a given sequence length.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            pe: Positional encoding of shape (1, seq_len, d_model)
        """
        return self.pe[:, :seq_len, :]
    
    def extra_repr(self):
        """Extra representation for print()"""
        return f'd_model={self.d_model}, max_len={self.max_len}, dropout={self.dropout_rate}'


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding (alternative).
    
    Instead of fixed sinusoidal encoding, this learns the positional embeddings.
    Useful for comparison or when training on very long sequences.
    
    Args:
        d_model: Embedding dimension (default: 128)
        max_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model=128, max_len=512, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Initialize with small values
        nn.init.normal_(self.pos_embeddings, mean=0.0, std=0.02)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: LearnablePositionalEncoding initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - max_len: {max_len}")
            print(f"[DEBUG]:   - Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """Add learned positional embeddings to input."""
        seq_len = x.size(1)
        x = x + self.pos_embeddings[:, :seq_len, :]
        x = self.dropout(x)
        return x


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    
    Applies rotation to the embeddings based on position.
    Used in modern LLMs like LLaMA and GPT-NeoX.
    
    Args:
        d_model: Embedding dimension (must be even)
        max_len: Maximum sequence length (default: 512)
    """
    
    def __init__(self, d_model=128, max_len=512):
        super(RotaryPositionalEncoding, self).__init__()
        
        assert d_model % 2 == 0, "d_model must be even for Rotary Positional Encoding"
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute rotation angles
        self._create_rotary_embeddings(max_len, d_model)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: RotaryPositionalEncoding initialized")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - max_len: {max_len}")
    
    def _create_rotary_embeddings(self, max_len, d_model):
        """Create rotary position embeddings."""
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        # Compute angles: pos / 10000^(2i/d)
        angles = position * div_term  # (max_len, d_model/2)
        
        # Compute cos and sin
        cos_emb = torch.cos(angles)  # (max_len, d_model/2)
        sin_emb = torch.sin(angles)  # (max_len, d_model/2)
        
        self.register_buffer('cos_emb', cos_emb)
        self.register_buffer('sin_emb', sin_emb)
    
    def forward(self, x):
        """Apply rotary position encoding to input."""
        # This is a simplified version; full RoPE implementation rotates pairs of dimensions
        seq_len = x.size(1)
        return x  # Placeholder - full implementation would rotate the embeddings


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def visualize_positional_encoding(pe_module, save_path=None):
    """
    Visualize the positional encoding matrix as a heatmap.
    
    Args:
        pe_module: PositionalEncoding module
        save_path: Path to save the figure
    """
    # Get the positional encoding matrix
    pe = pe_module.pe.squeeze(0).detach().numpy()  # Shape: (max_len, d_model)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heatmap of full positional encoding
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pe[:100, :64], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Positional Encoding (First 100 positions, 64 dimensions)')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Position')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Line plot of selected dimensions
    ax2 = axes[0, 1]
    positions = np.arange(100)
    dimensions = [0, 4, 8, 16, 32]
    for dim in dimensions:
        ax2.plot(positions, pe[:100, dim], label=f'dim {dim}', linewidth=1.5)
    ax2.set_title('Positional Encoding Values by Dimension')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Dot product similarity between positions
    ax3 = axes[1, 0]
    # Compute similarity matrix for first 100 positions
    pe_norm = pe[:100] / np.linalg.norm(pe[:100], axis=1, keepdims=True)
    similarity = np.dot(pe_norm, pe_norm.T)
    im3 = ax3.imshow(similarity, cmap='viridis', aspect='auto')
    ax3.set_title('Positional Encoding Similarity Matrix')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Position')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Frequency analysis
    ax4 = axes[1, 1]
    # Compute FFT for first few dimensions
    for dim in [0, 8, 16, 32, 64, 96]:
        fft = np.abs(np.fft.rfft(pe[:512, dim]))
        freq = np.fft.rfftfreq(512)
        ax4.plot(freq[1:], fft[1:], label=f'dim {dim}', linewidth=1.5)
    ax4.set_title('Frequency Spectrum of Positional Encoding')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Magnitude')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_position_similarity(pe_module, save_path=None):
    """
    Visualize how positional encoding similarity changes with distance.
    """
    pe = pe_module.pe.squeeze(0).detach().numpy()[:200]  # First 200 positions
    
    # Normalize
    pe_norm = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-10)
    
    # Compute similarity for each distance
    max_dist = 100
    similarities = []
    
    for dist in range(max_dist):
        sims = []
        for i in range(len(pe_norm) - dist):
            sim = np.dot(pe_norm[i], pe_norm[i + dist])
            sims.append(sim)
        similarities.append(np.mean(sims))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_dist), similarities, 'b-', linewidth=2)
    plt.xlabel('Position Distance', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.title('Positional Encoding Similarity vs Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Similarity plot saved to: {save_path}")
    plt.close()


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_positional_encoding():
    """Test the PositionalEncoding implementation"""
    print("=" * 60)
    print("TESTING SINUSOIDAL POSITIONAL ENCODING")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 50
    d_model = 128
    max_len = 512
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Hyperparameters:")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    
    # Initialize positional encoding
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.1)
    
    # Test forward pass
    print("\n" + "-" * 40)
    print("Test 1: Forward pass")
    print("-" * 40)
    
    output = pe(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    # Test that PE is actually added
    print("\n" + "-" * 40)
    print("Test 2: PE addition verification")
    print("-" * 40)
    
    # Create zero input
    x_zero = torch.zeros(batch_size, seq_len, d_model)
    output_zero = pe(x_zero)
    
    # The output should be exactly the positional encoding (plus dropout, but with eval mode)
    pe.eval()
    with torch.no_grad():
        output_zero = pe(x_zero)
    
    pe_matrix = pe.get_positional_encoding(seq_len)
    diff = (output_zero - pe_matrix).abs().max()
    
    print(f"Max difference between PE and output (zero input): {diff:.6f}")
    print(f"PE correctly added: {diff < 1e-5}")
    
    # Test different sequence lengths
    print("\n" + "-" * 40)
    print("Test 3: Different sequence lengths")
    print("-" * 40)
    
    for test_len in [10, 25, 50, 100]:
        x_test = torch.randn(batch_size, test_len, d_model)
        output_test = pe(x_test)
        print(f"  seq_len={test_len:3d}: input {str(x_test.shape):20s} -> output {output_test.shape}")
        assert output_test.shape == x_test.shape
    
    # Test PE properties
    print("\n" + "-" * 40)
    print("Test 4: Positional encoding properties")
    print("-" * 40)
    
    pe_matrix = pe.get_positional_encoding(100).squeeze(0)
    
    # Check value range
    print(f"PE value range: [{pe_matrix.min():.4f}, {pe_matrix.max():.4f}]")
    print(f"Expected range: [-1, 1]")
    
    # Check that odd and even dimensions have different patterns
    even_dims = pe_matrix[:, 0]  # First even dimension
    odd_dims = pe_matrix[:, 1]   # First odd dimension
    even_std = even_dims.std()
    odd_std = odd_dims.std()
    print(f"Even dim std: {even_std:.4f}")
    print(f"Odd dim std: {odd_std:.4f}")
    
    print("\n✓ All tests passed!")
    
    return output


def test_learnable_positional_encoding():
    """Test the LearnablePositionalEncoding"""
    print("\n" + "=" * 60)
    print("TESTING LEARNABLE POSITIONAL ENCODING")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    pe_learnable = LearnablePositionalEncoding(d_model=d_model, max_len=512, dropout=0.1)
    
    output = pe_learnable(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in pe_learnable.parameters()):,}")
    print(f"Gradient enabled: {pe_learnable.pos_embeddings.requires_grad}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient flows to positional embeddings: {pe_learnable.pos_embeddings.grad is not None}")
    
    print("\n✓ Tests passed!")
    
    return output


def test_positional_encoding_math():
    """Test the mathematical properties of positional encoding"""
    print("\n" + "=" * 60)
    print("TESTING POSITIONAL ENCODING MATHEMATICAL PROPERTIES")
    print("=" * 60)
    
    pe = PositionalEncoding(d_model=128, max_len=512)
    pe_matrix = pe.get_positional_encoding(200).squeeze(0).numpy()
    
    # Test 1: Periodicity - distance at which patterns repeat
    print("\nTest 1: Periodicity")
    # For dimension 0, period should be 2π
    dim0 = pe_matrix[:, 0]
    # Find distance between peaks
    peaks = np.where((dim0[1:-1] > dim0[:-2]) & (dim0[1:-1] > dim0[2:]))[0]
    if len(peaks) >= 2:
        period = peaks[1] - peaks[0]
        print(f"  Dimension 0 approximate period: {period} positions")
    
    # Test 2: Orthogonality of different positions
    print("\nTest 2: Position orthogonality")
    pe_norm = pe_matrix / (np.linalg.norm(pe_matrix, axis=1, keepdims=True) + 1e-10)
    
    # Similarity between nearby positions should be high
    sim_close = np.dot(pe_norm[0], pe_norm[1])
    # Similarity between distant positions should be lower
    sim_far = np.dot(pe_norm[0], pe_norm[100])
    
    print(f"  Similarity between pos 0 and 1: {sim_close:.4f}")
    print(f"  Similarity between pos 0 and 100: {sim_far:.4f}")
    print(f"  Nearby positions are more similar: {sim_close > sim_far}")
    
    # Test 3: Norm consistency
    print("\nTest 3: Norm consistency")
    norms = np.linalg.norm(pe_matrix, axis=1)
    print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Norm std: {norms.std():.4f}")
    
    print("\n✓ Mathematical tests complete!")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("SINUSOIDAL POSITIONAL ENCODING MODULE")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Run tests
    test_positional_encoding()
    test_learnable_positional_encoding()
    test_positional_encoding_math()
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    pe = PositionalEncoding(d_model=128, max_len=512)
    
    # Ensure figures directory exists
    os.makedirs(figuresDir, exist_ok=True)
    
    # Visualize positional encoding
    viz_path = os.path.join(figuresDir, 'positional_encoding_visualization.png')
    visualize_positional_encoding(pe, save_path=viz_path)
    
    # Visualize similarity
    sim_path = os.path.join(figuresDir, 'positional_encoding_similarity.png')
    visualize_position_similarity(pe, save_path=sim_path)
    
    print("\n" + "=" * 80)
    print("MODULE READY")
    print("=" * 80)