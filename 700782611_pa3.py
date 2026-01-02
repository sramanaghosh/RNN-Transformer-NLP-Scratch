"""
CS5720 - Assignment 3: RNNs and Transformers
Student Name: Sramana Ghosh
Student ID: 700782611
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from collections import defaultdict
import pickle

# ==================== Part 1: RNN Cells ====================

class VanillaRNNCell:
    """
    Vanilla RNN cell implementation.
    
    Implements the basic recurrent neural network cell with tanh activation:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize RNN cell parameters.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden state
            
        Note:
            Uses Xavier/He initialization for weights
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Xavier/He initialization for better gradient flow
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale_xh
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_h = np.zeros(hidden_dim)
        
        # Gradients
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass of RNN cell.
        
        Args:
            x: Input at current timestep [batch_size, input_dim]
            h_prev: Hidden state from previous timestep [batch_size, hidden_dim]
            
        Returns:
            h: Hidden state at current timestep [batch_size, hidden_dim]
            cache: Values needed for backward pass
        """
        # Compute linear combinations
        h_linear = np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
        
        # Apply tanh activation
        h = np.tanh(h_linear)
        
        # Cache values for backward pass
        cache = {
            'x': x,
            'h_prev': h_prev,
            'h': h,
            'h_linear': h_linear
        }
        
        return h, cache
    
    def backward(self, dh: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass of RNN cell.
        
        Args:
            dh: Gradient of loss w.r.t. hidden state [batch_size, hidden_dim]
            cache: Cache from forward pass
            
        Returns:
            dx: Gradient w.r.t. input [batch_size, input_dim]
            dh_prev: Gradient w.r.t. previous hidden state [batch_size, hidden_dim]
        """
        # Unpack cached values
        x = cache['x']
        h_prev = cache['h_prev']
        h = cache['h']
        h_linear = cache['h_linear']
        
        # Gradient through tanh
        dh_linear = dh * (1 - np.square(np.tanh(h_linear)))
        
        # Compute gradients
        dx = np.dot(dh_linear, self.W_xh.T)
        dh_prev = np.dot(dh_linear, self.W_hh.T)
        
        # Accumulate gradients for parameters
        self.dW_xh += np.dot(x.T, dh_linear)
        self.dW_hh += np.dot(h_prev.T, dh_linear)
        self.db_h += np.sum(dh_linear, axis=0)
        
        return dx, dh_prev


class LSTMCell:
    """
    LSTM (Long Short-Term Memory) cell implementation.
    
    Implements the LSTM architecture with forget, input, and output gates:
    f_t = sigmoid(W_xf @ x_t + W_hf @ h_{t-1} + b_f)
    i_t = sigmoid(W_xi @ x_t + W_hi @ h_{t-1} + b_i)
    c_tilde_t = tanh(W_xc @ x_t + W_hc @ h_{t-1} + b_c)
    c_t = f_t * c_{t-1} + i_t * c_tilde_t
    o_t = sigmoid(W_xo @ x_t + W_ho @ h_{t-1} + b_o)
    h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize LSTM cell parameters.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden state
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Xavier/He initialization
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Forget gate parameters
        self.W_xf = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hf = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_f = np.zeros(hidden_dim)  # Initialize forget gate bias to zeros
        
        # Input gate parameters
        self.W_xi = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hi = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_i = np.zeros(hidden_dim)
        
        # Cell candidate parameters
        self.W_xc = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hc = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_c = np.zeros(hidden_dim)
        
        # Output gate parameters
        self.W_xo = np.random.randn(input_dim, hidden_dim) * scale
        self.W_ho = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_o = np.zeros(hidden_dim)
        
        self._init_gradients()
        
    def _init_gradients(self):
        """Initialize gradient storage arrays for all parameters."""
        # Forget gate gradients
        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)
        
        # Input gate gradients
        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)
        
        # Cell candidate gradients
        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)
        
        # Output gate gradients
        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass of LSTM cell.
        
        Args:
            x: Input at current timestep [batch_size, input_dim]
            h_prev: Hidden state from previous timestep [batch_size, hidden_dim]
            c_prev: Cell state from previous timestep [batch_size, hidden_dim]
            
        Returns:
            h: Hidden state at current timestep [batch_size, hidden_dim]
            c: Cell state at current timestep [batch_size, hidden_dim]
            cache: Values needed for backward pass
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
        
        # Forget gate
        f_gate = sigmoid(np.dot(x, self.W_xf) + np.dot(h_prev, self.W_hf) + self.b_f)
        
        # Input gate
        i_gate = sigmoid(np.dot(x, self.W_xi) + np.dot(h_prev, self.W_hi) + self.b_i)
        
        # Cell candidate
        c_tilde = np.tanh(np.dot(x, self.W_xc) + np.dot(h_prev, self.W_hc) + self.b_c)
        
        # Cell state update
        c = f_gate * c_prev + i_gate * c_tilde
        
        # Output gate
        o_gate = sigmoid(np.dot(x, self.W_xo) + np.dot(h_prev, self.W_ho) + self.b_o)
        
        # Hidden state
        h = o_gate * np.tanh(c)
        
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f': f_gate, 'i': i_gate, 'g': c_tilde,
            'o': o_gate, 'next_c': c, 'next_h': h,
            'Wx': np.concatenate([self.W_xf, self.W_xi, self.W_xc, self.W_xo], axis=1),
            'Wh': np.concatenate([self.W_hf, self.W_hi, self.W_hc, self.W_ho], axis=1),
            'b': np.concatenate([self.b_f, self.b_i, self.b_c, self.b_o])
        }
        
        return h, c, cache
    
    def backward(self, dh: np.ndarray, dc: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of LSTM cell.
        
        Args:
            dh: Gradient of loss w.r.t. hidden state [batch_size, hidden_dim]
            dc: Gradient of loss w.r.t. cell state [batch_size, hidden_dim]
            cache: Cache from forward pass
            
        Returns:
            dx: Gradient w.r.t. input [batch_size, input_dim]
            dh_prev: Gradient w.r.t. previous hidden state [batch_size, hidden_dim]
            dc_prev: Gradient w.r.t. previous cell state [batch_size, hidden_dim]
        """
        # Return the results from lstm_step_backward
        dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dh, dc, cache)
        # Update gradients
        self.dW_xf += dWx[:, :self.hidden_dim]
        self.dW_xi += dWx[:, self.hidden_dim:2*self.hidden_dim]
        self.dW_xc += dWx[:, 2*self.hidden_dim:3*self.hidden_dim]
        self.dW_xo += dWx[:, 3*self.hidden_dim:]
        
        self.dW_hf += dWh[:, :self.hidden_dim]
        self.dW_hi += dWh[:, self.hidden_dim:2*self.hidden_dim]
        self.dW_hc += dWh[:, 2*self.hidden_dim:3*self.hidden_dim]
        self.dW_ho += dWh[:, 3*self.hidden_dim:]
        
        self.db_f += db[:self.hidden_dim]
        self.db_i += db[self.hidden_dim:2*self.hidden_dim]
        self.db_c += db[2*self.hidden_dim:3*self.hidden_dim]
        self.db_o += db[3*self.hidden_dim:]
        
        return dx, dprev_h, dprev_c


class GRUCell:
    """
    GRU (Gated Recurrent Unit) cell implementation.
    
    Implements the GRU architecture with reset and update gates:
    r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1} + b_r)
    z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1} + b_z)
    h_tilde_t = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}) + b_h)
    h_t = z_t * h_{t-1} + (1 - z_t) * h_tilde_t
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize GRU cell parameters.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden state
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Xavier/He initialization
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Reset gate parameters
        self.W_xr = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hr = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        
        # Update gate parameters
        self.W_xz = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hz = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_z = np.zeros(hidden_dim)
        
        # Candidate parameters
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)
        
        self._init_gradients()
        
    def _init_gradients(self):
        """Initialize gradient storage arrays for all parameters."""
        # Reset gate gradients
        self.dW_xr = np.zeros_like(self.W_xr)
        self.dW_hr = np.zeros_like(self.W_hr)
        self.db_r = np.zeros_like(self.b_r)
        
        # Update gate gradients
        self.dW_xz = np.zeros_like(self.W_xz)
        self.dW_hz = np.zeros_like(self.W_hz)
        self.db_z = np.zeros_like(self.b_z)
        
        # Candidate gradients
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass of GRU cell.
        
        Args:
            x: Input at current timestep [batch_size, input_dim]
            h_prev: Hidden state from previous timestep [batch_size, hidden_dim]
            
        Returns:
            h: Hidden state at current timestep [batch_size, hidden_dim]
            cache: Values needed for backward pass
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
        
        # Reset gate
        r = sigmoid(np.dot(x, self.W_xr) + np.dot(h_prev, self.W_hr) + self.b_r)
        
        # Update gate
        z = sigmoid(np.dot(x, self.W_xz) + np.dot(h_prev, self.W_hz) + self.b_z)
        
        # Candidate state
        h_reset = r * h_prev
        h_tilde = np.tanh(np.dot(x, self.W_xh) + np.dot(h_reset, self.W_hh) + self.b_h)
        
        # New hidden state
        h = z * h_prev + (1 - z) * h_tilde
        
        cache = {
            'x': x, 'h_prev': h_prev,
            'r': r, 'z': z, 'h_tilde': h_tilde,
            'h_reset': h_reset
        }
        
        return h, cache
    
    def backward(self, dh: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass of GRU cell.
        
        Args:
            dh: Gradient of loss w.r.t. hidden state [batch_size, hidden_dim]
            cache: Cache from forward pass
            
        Returns:
            dx: Gradient w.r.t. input [batch_size, input_dim]
            dh_prev: Gradient w.r.t. previous hidden state [batch_size, hidden_dim]
        """
        x, h_prev = cache['x'], cache['h_prev']
        r, z = cache['r'], cache['z']
        h_tilde = cache['h_tilde']
        h_reset = cache['h_reset']
        
        # Gradient through mixing
        dh_tilde = dh * (1 - z)
        dh_prev_from_mix = dh * z
        
        # Gradient through tanh
        dh_linear = dh_tilde * (1 - h_tilde ** 2)
        
        # Gradients for candidate computation
        dx_h = np.dot(dh_linear, self.W_xh.T)
        dh_reset = np.dot(dh_linear, self.W_hh.T)
        
        # Gradient through reset gate
        dr = dh_reset * h_prev
        dh_prev_from_reset = dh_reset * r
        
        # Reset gate gradients
        dr_input = dr * r * (1 - r)
        dx_r = np.dot(dr_input, self.W_xr.T)
        dh_prev_r = np.dot(dr_input, self.W_hr.T)
        
        # Update gate gradients
        dz = dh * (h_prev - h_tilde)
        dz_input = dz * z * (1 - z)
        dx_z = np.dot(dz_input, self.W_xz.T)
        dh_prev_z = np.dot(dz_input, self.W_hz.T)
        
        # Combine gradients
        dx = dx_h + dx_r + dx_z
        dh_prev = dh_prev_from_mix + dh_prev_from_reset + dh_prev_r + dh_prev_z
        
        # Accumulate parameter gradients
        self.dW_xh += np.dot(x.T, dh_linear)
        self.dW_hh += np.dot(h_reset.T, dh_linear)
        self.db_h += np.sum(dh_linear, axis=0)
        
        self.dW_xr += np.dot(x.T, dr_input)
        self.dW_hr += np.dot(h_prev.T, dr_input)
        self.db_r += np.sum(dr_input, axis=0)
        
        self.dW_xz += np.dot(x.T, dz_input)
        self.dW_hz += np.dot(h_prev.T, dz_input)
        self.db_z += np.sum(dz_input, axis=0)
        
        return dx, dh_prev


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ultra-optimized scaled dot-product attention implementation with zero-copy operations.
    Uses pre-allocated buffers and minimal memory operations for maximum speed.
    
    Args:
        Q: Query matrix [batch_size * num_heads, seq_len_q, d_k]
        K: Key matrix [batch_size * num_heads, seq_len_k, d_k]
        V: Value matrix [batch_size * num_heads, seq_len_v, d_v]
        mask: Optional mask [batch_size, seq_len_q, seq_len_k]
            Mask values of 1 indicate positions to mask out
        
    Returns:
        output: [batch_size * num_heads, seq_len_q, d_v]
        attention_weights: [batch_size * num_heads, seq_len_q, seq_len_k]
    """
    # Get dimensions once
    batch_heads, seq_len, d_k = Q.shape
    
    # Pre-compute scaling and apply to Q for efficient matmul
    scaling = 1.0 / np.sqrt(d_k)
    Q_scaled = Q * scaling  # This will use broadcasting, more efficient than division
    
    # Compute attention scores efficiently
    scores = np.matmul(Q_scaled, K.transpose(0, 2, 1))  # [batch_heads, seq_len_q, seq_len_k]
    
    # Handle masking efficiently
    if mask is not None:
        # Extract batch size from mask and compute num_heads
        batch_size = mask.shape[0]
        num_heads = batch_heads // batch_size
        
        # Reshape scores for masking if using multiple heads
        scores = scores.reshape(batch_size, num_heads, seq_len, seq_len)
        
        # Expand mask for broadcasting if needed
        if len(mask.shape) == 3:
            mask = mask[:, None]  # [batch, 1, seq_len, seq_len]
        
        # Apply mask - using masked positions (1s) to set attention to -inf
        scores = np.where(mask > 0, -np.inf, scores)
        
        # Reshape back to original shape for softmax
        scores = scores.reshape(batch_heads, seq_len, seq_len)
    
    # Optimized softmax with enhanced numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
    
    # Compute attention output in one step
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights




class MultiHeadSelfAttention:
    """Ultra-optimized multi-head self-attention implementation."""
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention with optimized memory layout.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Initialize weight matrices with optimal memory layout and scaling
        scale = np.sqrt(2.0 / d_model)
        # Use F-contiguous layout for more efficient matrix operations
        self.W_q = np.asfortranarray(np.random.randn(d_model, d_model) * scale)
        self.W_k = np.asfortranarray(np.random.randn(d_model, d_model) * scale)
        self.W_v = np.asfortranarray(np.random.randn(d_model, d_model) * scale)
        self.W_o = np.asfortranarray(np.random.randn(d_model, d_model) * scale)
        
        # Pre-allocate intermediate buffers
        self._temp_qkv = None
        self._temp_context = None
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-optimized forward pass with minimal memory operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize or resize intermediate buffers if needed
        if (self._temp_qkv is None or 
            self._temp_qkv.shape != (3, batch_size, seq_len, self.num_heads, self.head_dim)):
            self._temp_qkv = np.empty((3, batch_size, seq_len, self.num_heads, self.head_dim))
            self._temp_context = np.empty((batch_size, seq_len, self.d_model))
        
        # Project Q, K, V in parallel with minimal reshaping
        # Using F-contiguous arrays for better matrix multiplication performance
        q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Efficient reshape and transpose using pre-allocated buffers
        q = q.transpose(0, 2, 1, 3).reshape(-1, seq_len, self.head_dim)
        k = k.transpose(0, 2, 1, 3).reshape(-1, seq_len, self.head_dim)
        v = v.transpose(0, 2, 1, 3).reshape(-1, seq_len, self.head_dim)
        
        # Apply scaled dot-product attention
        context, attn = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape output efficiently using pre-allocated buffer
        context = context.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final projection with F-contiguous array
        output = np.dot(context, self.W_o, out=self._temp_context)
        
        # Reshape attention weights for return
        attention_weights = attn.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        return output, attention_weights



def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate positional encoding for transformer models.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        pos_encoding: Positional encoding matrix [seq_len, d_model]
    """
    pos_encoding = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding


# ==================== Part 3: Sequence Models ====================

class RNNLanguageModel:
    """
    RNN-based language model for text generation.
    
    Combines embedding layer, RNN cell (vanilla/LSTM/GRU), and output projection
    for character or word-level language modeling.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, cell_type: str = 'vanilla'):
        """
        Initialize language model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            cell_type: Type of RNN cell ('vanilla', 'lstm', 'gru')
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        
        # Initialize embedding layer
        scale = np.sqrt(2.0 / (vocab_size + hidden_dim))
        self.embedding = np.random.randn(vocab_size, hidden_dim) * scale
        
        # Initialize RNN cell
        if cell_type == 'vanilla':
            self.cell = VanillaRNNCell(hidden_dim, hidden_dim)
        elif cell_type == 'lstm':
            self.cell = LSTMCell(hidden_dim, hidden_dim)
        elif cell_type == 'gru':
            self.cell = GRUCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
            
        # Initialize output projection
        self.output_proj = np.random.randn(hidden_dim, vocab_size) * scale
        self.b_proj = np.zeros(vocab_size)
    
    def forward(self, input_ids: np.ndarray, initial_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through language model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            initial_state: Initial hidden state (optional)
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize states
        if initial_state is None:
            h = np.zeros((batch_size, self.hidden_dim))
            if self.cell_type == 'lstm':
                c = np.zeros((batch_size, self.hidden_dim))
        else:
            h = initial_state
            if self.cell_type == 'lstm':
                c = np.zeros_like(h)
        
        # Storage for outputs
        logits = np.zeros((batch_size, seq_len, self.vocab_size))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_dim))
        
        # Process sequence
        for t in range(seq_len):
            # Embed current input
            x_t = self.embedding[input_ids[:, t]]
            
            # RNN step
            if self.cell_type == 'lstm':
                h, c, _ = self.cell.forward(x_t, h, c)
            else:
                h, _ = self.cell.forward(x_t, h)
            
            # Store hidden state
            hidden_states[:, t] = h
            
            # Project to vocabulary
            logits[:, t] = np.dot(h, self.output_proj) + self.b_proj
        
        return logits, hidden_states
    
    def generate(self, prompt_ids: np.ndarray, max_length: int, temperature: float = 1.0) -> np.ndarray:
        """
        Generate text from prompt.
        
        Args:
            prompt_ids: Prompt token IDs [1, prompt_len]
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            generated_ids: Generated token IDs [1, max_length]
        """
        generated_ids = np.zeros((1, max_length), dtype=int)
        generated_ids[0, :prompt_ids.shape[1]] = prompt_ids
        
        # Initialize state with prompt
        _, hidden_states = self.forward(prompt_ids)
        h = hidden_states[:, -1]  # Take last hidden state
        
        # Initialize cell state for LSTM
        if self.cell_type == 'lstm':
            c = np.zeros_like(h)
        
        # Generate new tokens
        for t in range(prompt_ids.shape[1], max_length):
            # Get last generated token
            last_token = generated_ids[0, t-1]
            x_t = self.embedding[last_token]
            
            # RNN step
            if self.cell_type == 'lstm':
                h, c, _ = self.cell.forward(x_t, h, c)
            else:
                h, _ = self.cell.forward(x_t, h)
            
            # Get logits and apply temperature
            logits = (np.dot(h, self.output_proj) + self.b_proj) / temperature
            
            # Sample from distribution
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(self.vocab_size, p=probs.ravel())
            
            generated_ids[0, t] = next_token
        
        return generated_ids


class BiLSTMClassifier:
    """
    Bidirectional LSTM classifier for sequence classification tasks.
    
    Features:
    - Bidirectional processing with improved LSTM cells
    - Advanced attention mechanism
    - Enhanced embedding with dropout
    - Layer normalization
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int, use_attention: bool = True):
        """
        Initialize classifier with improved architecture.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            num_classes: Number of output classes
            use_attention: Whether to use attention mechanism
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Enhanced embedding initialization
        scale = np.sqrt(1.0 / hidden_dim)  # Adjusted scale for better gradient flow
        self.embedding = np.random.uniform(-scale, scale, (vocab_size, hidden_dim))
        
        # Improved bidirectional LSTM with larger capacity
        lstm_hidden = hidden_dim * 2  # Increased capacity
        self.lstm_forward = LSTMCell(hidden_dim, lstm_hidden)
        self.lstm_backward = LSTMCell(hidden_dim, lstm_hidden)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(lstm_hidden * 2)  # For concatenated states
        self.beta1 = np.zeros(lstm_hidden * 2)
        self.gamma2 = np.ones(num_classes)  # For classifier output
        self.beta2 = np.zeros(num_classes)
        
        # Enhanced attention mechanism
        if use_attention:
            attn_dim = lstm_hidden * 2
            self.attention_query = np.random.randn(attn_dim, attn_dim) * np.sqrt(1.0 / attn_dim)
            self.attention_key = np.random.randn(attn_dim, attn_dim) * np.sqrt(1.0 / attn_dim)
            self.attention_value = np.random.randn(attn_dim, attn_dim) * np.sqrt(1.0 / attn_dim)
        
        # Improved classifier initialization
        classifier_scale = np.sqrt(2.0 / (lstm_hidden * 2))
        self.classifier = np.random.randn(lstm_hidden * 2, num_classes) * classifier_scale
        self.b_classifier = np.zeros(num_classes)
        
        # Dropout rates
        self.embed_dropout = 0.2
        self.lstm_dropout = 0.3
        self.attention_dropout = 0.2
        self.classifier_dropout = 0.3
    
    def forward(self, input_ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Forward pass through classifier with improved architecture.
        
        Args:
            input_ids: Input token IDs [batch_size, max_seq_len]
            lengths: Actual sequence lengths [batch_size]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, max_seq_len = input_ids.shape
        is_training = True  # Set to True during training, False during inference
        
        # Embed inputs with dropout
        embedded = self.embedding[input_ids]
        if is_training:
            dropout_mask = (np.random.rand(*embedded.shape) > self.embed_dropout) / (1 - self.embed_dropout)
            embedded = embedded * dropout_mask
        
        # Forward LSTM with increased capacity
        h_forward = np.zeros((batch_size, self.hidden_dim * 2))
        c_forward = np.zeros((batch_size, self.hidden_dim * 2))
        forward_states = np.zeros((batch_size, max_seq_len, self.hidden_dim * 2))
        
        for t in range(max_seq_len):
            # Apply LSTM dropout during training
            if is_training:
                lstm_mask = (np.random.rand(*embedded[:, t].shape) > self.lstm_dropout) / (1 - self.lstm_dropout)
                lstm_input = embedded[:, t] * lstm_mask
            else:
                lstm_input = embedded[:, t]
            
            h_forward, c_forward, _ = self.lstm_forward.forward(
                lstm_input, h_forward, c_forward
            )
            forward_states[:, t] = h_forward
        
        # Backward LSTM with increased capacity
        h_backward = np.zeros((batch_size, self.hidden_dim * 2))
        c_backward = np.zeros((batch_size, self.hidden_dim * 2))
        backward_states = np.zeros((batch_size, max_seq_len, self.hidden_dim * 2))
        
        for t in range(max_seq_len - 1, -1, -1):
            # Apply LSTM dropout during training
            if is_training:
                lstm_mask = (np.random.rand(*embedded[:, t].shape) > self.lstm_dropout) / (1 - self.lstm_dropout)
                lstm_input = embedded[:, t] * lstm_mask
            else:
                lstm_input = embedded[:, t]
            
            h_backward, c_backward, _ = self.lstm_backward.forward(
                lstm_input, h_backward, c_backward
            )
            backward_states[:, t] = h_backward
        
        # Combine directions and apply layer normalization
        hidden_states = np.concatenate([forward_states, backward_states], axis=-1)
        
        # Layer normalization
        mean = np.mean(hidden_states, axis=-1, keepdims=True)
        var = np.var(hidden_states, axis=-1, keepdims=True) + 1e-5
        normalized = (hidden_states - mean) / np.sqrt(var)
        hidden_states = self.gamma1 * normalized + self.beta1
        
        # Enhanced attention mechanism
        if self.use_attention:
            # Compute Q, K, V with learned projections
            Q = np.dot(hidden_states, self.attention_query)
            K = np.dot(hidden_states, self.attention_key)
            V = np.dot(hidden_states, self.attention_value)
            
            # Scale dot-product attention
            attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(K.shape[-1])
            
            # Create and apply attention mask
            mask = np.zeros((batch_size, max_seq_len, max_seq_len))
            for i, length in enumerate(lengths):
                mask[i, :length, :length] = 1
            
            attention_scores = np.where(mask == 0, -1e9, attention_scores)
            
            # Softmax and dropout
            attention_probs = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
            attention_probs /= (np.sum(attention_probs, axis=-1, keepdims=True) + 1e-9)
            if is_training:
                attention_probs = attention_probs * (np.random.rand(*attention_probs.shape) > self.attention_dropout) / (1 - self.attention_dropout)
            
            # Compute weighted sum
            hidden_states = np.matmul(attention_probs, V)
            
            # Store hidden states for later use
            self.hidden_states = hidden_states.copy()
            
            # Get sentence representation using weighted average of states
            weights = np.zeros((batch_size, max_seq_len))
            for i, length in enumerate(lengths):
                weights[i, :length] = 1 / length
            weights = weights.reshape(batch_size, max_seq_len, 1)
            hidden_states = np.sum(hidden_states * weights, axis=1)
        else:
            # Store hidden states for later use
            self.hidden_states = hidden_states.copy()
            
            # Use a combination of max and average pooling
            max_pooled = np.max(hidden_states, axis=1)
            avg_pooled = np.mean(hidden_states, axis=1)
            hidden_states = np.concatenate([max_pooled, avg_pooled], axis=-1) / 2
        
        # Apply dropout before classification
        if is_training:
            hidden_states = hidden_states * (np.random.rand(*hidden_states.shape) > self.classifier_dropout) / (1 - self.classifier_dropout)
        
        # Classification with layer normalization
        logits = np.dot(hidden_states, self.classifier) + self.b_classifier
        
        # Final layer normalization
        mean = np.mean(logits, axis=-1, keepdims=True)
        var = np.var(logits, axis=-1, keepdims=True) + 1e-5
        normalized = (logits - mean) / np.sqrt(var)
        logits = self.gamma2 * normalized + self.beta2
        
        return logits

    def create_attention_mask(self, batch_size, max_length):
        mask = np.zeros((batch_size, max_length, max_length))
        for i, length in enumerate(self.lengths):
            mask[i, :length, :length] = 1
        return mask

    def compute_attention(self, hidden_states):
        attention_scores = np.dot(hidden_states, hidden_states.transpose(0, 2, 1))
        attention_scores /= np.sqrt(hidden_states.shape[-1])  # Scale by sqrt(d_k)
        
        # Apply attention mask
        mask = self.create_attention_mask(hidden_states.shape[0], hidden_states.shape[1])
        attention_scores = np.where(mask == 0, -1e9, attention_scores)
        
        # Softmax
        attention_probs = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_probs /= (np.sum(attention_probs, axis=-1, keepdims=True) + 1e-9)
        
        # Apply dropout during training
        if hasattr(self, 'is_training') and self.is_training:
            attention_probs = attention_probs * (np.random.rand(*attention_probs.shape) > self.attention_dropout) / (1 - self.attention_dropout)
        
        # Compute attention-weighted values
        attention_output = np.matmul(attention_probs, hidden_states)
        
        # Apply attention mask for proper masking
        if hasattr(self, 'attention_mask') and self.attention_mask is not None:
            attention_output = attention_output * self.attention_mask

        return attention_output, attention_probs
        
        # Compute logits with improved numerical stability
        logits = np.dot(final_state, self.classifier) + self.b_classifier
        
        # Ensure logits are well-scaled
        logits = logits - np.max(logits, axis=-1, keepdims=True)  # For numerical stability
        
        return logits


class TextProcessor:
    """
    Text processing utilities for sequence modeling.
    
    Features:
    - Character or word-level tokenization
    - Vocabulary management
    - Batch creation with padding
    """
    
    def __init__(self, level: str = 'char'):
        """
        Initialize text processor.
        
        Args:
            level: Tokenization level ('char' or 'word')
        """
        self.level = level
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for inclusion
        """
        # Count tokens
        token_counts = defaultdict(int)
        for text in texts:
            tokens = list(text) if self.level == 'char' else text.split()
            for token in tokens:
                token_counts[token] += 1
        
        # Add frequent tokens to vocabulary
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.reverse_vocab[self.vocab[token]] = token
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            token_ids: Array of token IDs
        """
        if self.level == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
            
        return np.array([self.vocab.get(token, self.vocab['<UNK>']) 
                        for token in tokens])
    
    def decode(self, token_ids: np.ndarray) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: Array of token IDs
            
        Returns:
            text: Decoded text
        """
        tokens = [self.reverse_vocab.get(int(idx), '<UNK>') for idx in token_ids]
        if self.level == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def create_batches(self, texts: List[str], batch_size: int, max_length: Optional[int] = None) -> List[Dict]:
        """
        Create batched data for training.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            max_length: Maximum sequence length (optional)
            
        Returns:
            batches: List of batch dictionaries
        """
        # Encode all texts
        encoded_texts = [self.encode(text) for text in texts]
        
        # Determine max length if not provided
        if max_length is None:
            max_length = max(len(text) for text in encoded_texts)
        
        # Create batches
        batches = []
        for i in range(0, len(encoded_texts), batch_size):
            batch_texts = encoded_texts[i:i + batch_size]
            
            # Get actual lengths
            lengths = np.array([len(text) for text in batch_texts])
            
            # Pad sequences
            padded = np.zeros((len(batch_texts), max_length), dtype=int)
            for j, text in enumerate(batch_texts):
                padded[j, :len(text)] = text[:max_length]
            
            # Create attention mask
            mask = np.zeros((len(batch_texts), max_length))
            for j, length in enumerate(lengths):
                mask[j, :min(length, max_length)] = 1
            
            batches.append({
                'input_ids': padded,
                'lengths': lengths,
                'mask': mask  # Changed from attention_mask to mask to match test requirements
            })
        
        return batches


def truncated_bptt(loss_fn, model, sequences, k1: int = 20, k2: int = 5):
    """
    Implement Truncated Backpropagation Through Time.
    
    Args:
        loss_fn: Loss function
        model: RNN model
        sequences: Input sequences
        k1: Forward pass truncation length
        k2: Backward pass truncation length
    """
    batch_size, seq_len = sequences.shape
    
    # Initialize hidden state
    h = np.zeros((batch_size, model.hidden_dim))
    if hasattr(model, 'cell') and model.cell_type == 'lstm':
        c = np.zeros((batch_size, model.hidden_dim))
    
    # Process sequence in chunks
    for t in range(0, seq_len - k1, k2):
        # Forward pass through chunk
        chunk = sequences[:, t:t+k1]
        if hasattr(model, 'cell') and model.cell_type == 'lstm':
            logits, (h, c) = model.forward(chunk, (h.detach(), c.detach()))
        else:
            logits, h = model.forward(chunk, h.detach())
        
        # Compute loss and backward pass
        loss = loss_fn(logits, sequences[:, t+1:t+k1+1])
        loss.backward()
        
        # Detach hidden state for next chunk
        h = h.detach()
        if hasattr(model, 'cell') and model.cell_type == 'lstm':
            c = c.detach()


def gradient_clipping(gradients: Dict[str, np.ndarray], max_norm: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum gradient norm
        
    Returns:
        clipped_gradients: Clipped gradients
    """
    # Compute global norm
    global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients.values()))
    
    # Clip if necessary
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-6)
        clipped_gradients = {k: v * scale for k, v in gradients.items()}
        return clipped_gradients
    
    return gradients


def lstm_step_backward(dnext_h: np.ndarray, dnext_c: np.ndarray, cache: Dict) -> Tuple:
    """
    Backward pass for a single timestep of an LSTM.
    
    Args:
        dnext_h: Gradients of next hidden state, of shape (N, H)
        dnext_c: Gradients of next cell state, of shape (N, H)
        cache: Cache object from the forward pass
        
    Returns:
        dx: Gradient of input data, of shape (N, D)
        dprev_h: Gradient of previous hidden state, of shape (N, H)
        dprev_c: Gradient of previous cell state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        db: Gradient of biases, of shape (4H,)
    """
    # Unpack cache
    x, prev_h, prev_c = cache['x'], cache['h_prev'], cache['c_prev']
    i, f, o, g = cache['i'], cache['f'], cache['o'], cache['g']
    Wx, Wh, b = cache['Wx'], cache['Wh'], cache['b']
    next_c, next_h = cache['next_c'], cache['next_h']
    
    # Initialize gradients
    dx, dprev_h, dprev_c = None, None, None
    dWx, dWh, db = None, None, None
    
    # Dimensions
    N, H = dnext_h.shape
    D = x.shape[1]
    
    # Backprop through the composed layer
    do = dnext_h * np.tanh(next_c)
    dnext_c = dnext_c + dnext_h * o * (1 - np.tanh(next_c)**2)
    
    # Backprop through internal gates
    di = dnext_c * g
    df = dnext_c * prev_c
    dg = dnext_c * i
    dprev_c = dnext_c * f
    
    # Backprop through nonlinearities
    di = di * i * (1 - i)  # sigmoid derivative
    df = df * f * (1 - f)  # sigmoid derivative
    do = do * o * (1 - o)  # sigmoid derivative
    dg = dg * (1 - g**2)   # tanh derivative
    
    # Concatenate gates
    dactivations = np.concatenate((di, df, do, dg), axis=1)
    
    # Compute gradients for weights and biases
    dx = np.dot(dactivations, Wx.T)
    dprev_h = np.dot(dactivations, Wh.T)
    dWx = np.dot(x.T, dactivations)
    dWh = np.dot(prev_h.T, dactivations)
    db = np.sum(dactivations, axis=0)
    
    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_step_backward(dnext_h: np.ndarray, dnext_c: np.ndarray, cache: Dict) -> Tuple:
    """
    Backward pass for a single timestep of an LSTM.
    
    Args:
        dnext_h: Gradients of next hidden state, of shape (N, H)
        dnext_c: Gradients of next cell state, of shape (N, H)
        cache: Cache object from the forward pass
        
    Returns:
        dx: Gradient of input data, of shape (N, D)
        dprev_h: Gradient of previous hidden state, of shape (N, H)
        dprev_c: Gradient of previous cell state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        db: Gradient of biases, of shape (4H,)
    """
    # Unpack cache
    x, prev_h, prev_c = cache['x'], cache['h_prev'], cache['c_prev']
    i, f, o, g = cache['i'], cache['f'], cache['o'], cache['g']
    Wx, Wh, b = cache['Wx'], cache['Wh'], cache['b']
    next_c, next_h = cache['next_c'], cache['next_h']
    
    # Initialize gradients
    N, H = dnext_h.shape
    D = x.shape[1]
    
    # Backprop through the composed layer
    do = dnext_h * np.tanh(next_c)
    dnext_c = dnext_c + dnext_h * o * (1 - np.tanh(next_c)**2)
    
    # Backprop through internal gates
    di = dnext_c * g
    df = dnext_c * prev_c
    dg = dnext_c * i
    dprev_c = dnext_c * f
    
    # Backprop through nonlinearities
    di_input = di * i * (1 - i)  # sigmoid derivative
    df_input = df * f * (1 - f)  # sigmoid derivative
    do_input = do * o * (1 - o)  # sigmoid derivative
    dg_input = dg * (1 - g**2)   # tanh derivative
    
    # Concatenate gates
    dactivations = np.concatenate((di_input, df_input, do_input, dg_input), axis=1)
    
    # Compute gradients for weights and biases
    dx = np.dot(dactivations, Wx.T)
    dprev_h = np.dot(dactivations, Wh.T)
    dWx = np.dot(x.T, dactivations)
    dWh = np.dot(prev_h.T, dactivations)
    db = np.sum(dactivations, axis=0)
    
    return dx, dprev_h, dprev_c, dWx, dWh, db

def compute_perplexity(logits: np.ndarray, targets: np.ndarray, lengths: np.ndarray) -> float:
    """
    Compute perplexity for language modeling.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        lengths: Actual sequence lengths [batch_size]
        
    Returns:
        perplexity: Perplexity score
    """
    # Convert logits to probabilities
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    
    # Compute negative log likelihood
    nll = 0
    total_tokens = 0
    
    for i in range(len(lengths)):
        for t in range(lengths[i]):
            nll -= np.log(probs[i, t, targets[i, t]] + 1e-10)
            total_tokens += 1
    
    # Compute perplexity
    perplexity = np.exp(nll / total_tokens)
    
    return perplexity


def compute_bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
    """
    Compute BLEU score for generated text.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        n: Maximum n-gram order
        
    Returns:
        bleu_score: BLEU score
    """
    def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Helper function to get n-gram counts."""
        ngram_counts = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_counts[ngram] += 1
        return ngram_counts
    
    total_score = 0
    weights = [1/n] * n
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Compute modified precision for each n-gram order
        scores = []
        for i in range(n):
            pred_ngrams = get_ngrams(pred_tokens, i+1)
            ref_ngrams = get_ngrams(ref_tokens, i+1)
            
            # Count matches
            matches = 0
            total = 0
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams[ngram])
                total += count
            
            scores.append(matches / (total + 1e-10))
        
        # Compute brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        # Compute final score for this pair
        score = bp * np.exp(np.sum([w * np.log(s + 1e-10) for w, s in zip(weights, scores)]))
        total_score += score
    
    return total_score / len(predictions)


def train_language_model(model: RNNLanguageModel, train_data: List[str], 
                        val_data: List[str], config: Dict) -> Dict:
    """
    Train language model.
    
    Args:
        model: Language model
        train_data: Training texts
        val_data: Validation texts
        config: Training configuration
        
    Returns:
        results: Training results and metrics
    """
    # Create text processor and build vocab
    text_processor = TextProcessor(level=config.get('level', 'char'))
    text_processor.build_vocab(train_data, min_freq=config.get('min_freq', 1))
    
    # Training parameters
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 10)
    learning_rate = config.get('learning_rate', 0.0005)
    
    # Initialize results tracking
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': [],
        'generated_samples': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Create training batches
        train_batches = text_processor.create_batches(train_data, batch_size)
        
        # Train on batches
        epoch_loss = 0
        for batch in train_batches:
            # Forward pass
            logits, _ = model.forward(batch['input_ids'])
            
            # Compute loss and perplexity
            perplexity = compute_perplexity(logits, batch['input_ids'][:, 1:], batch['lengths'] - 1)
            epoch_loss += perplexity
            
            # Backward pass with truncated BPTT
            truncated_bptt(compute_perplexity, model, batch['input_ids'])
            
            # Update parameters
            gradients = {
                'embedding': model.embedding,
                'output_proj': model.output_proj,
                'b_proj': model.b_proj
            }
            gradients = gradient_clipping(gradients)
            
            for param_name, grad in gradients.items():
                setattr(model, param_name, 
                        getattr(model, param_name) - learning_rate * grad)
        
        # Validation
        val_batches = text_processor.create_batches(val_data, batch_size)
        val_loss = 0
        for batch in val_batches:
            logits, _ = model.forward(batch['input_ids'])
            val_loss += compute_perplexity(logits, batch['input_ids'][:, 1:], batch['lengths'] - 1)
        
        # Generate sample
        prompt = val_data[0][:50]  # Use first 50 chars as prompt
        generated = model.generate(
            text_processor.encode(prompt)[None, :],
            max_length=100,
            temperature=0.8
        )
        generated_text = text_processor.decode(generated[0])
        
        # Update results
        results['train_loss'].append(epoch_loss / len(train_batches))
        results['val_loss'].append(val_loss / len(val_batches))
        results['generated_samples'].append(generated_text)
    
    return results


def train_classifier(model: BiLSTMClassifier, train_data: List[Tuple[str, int]], 
                    val_data: List[Tuple[str, int]], config: Dict) -> Dict:
    """
    Train sentiment classifier.
    
    Args:
        model: Classifier model
        train_data: Training data (text, label) pairs
        val_data: Validation data
        config: Training configuration
        
    Returns:
        results: Training results and metrics
    """
    # Create text processor
    text_processor = TextProcessor(level='word')
    texts = [text for text, _ in train_data]
    text_processor.build_vocab(texts, min_freq=config.get('min_freq', 2))
    
    # Training parameters
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 10)
    learning_rate = config.get('learning_rate', 0.0005)
    
    # Initialize results
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'confusion_matrix': None
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Prepare batches
        indices = np.random.permutation(len(train_data))
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            texts = [train_data[j][0] for j in batch_indices]
            labels = np.array([train_data[j][1] for j in batch_indices])
            
            # Process batch
            batch = text_processor.create_batches(texts, batch_size)[0]
            
            # Forward pass
            logits = model.forward(batch['input_ids'], batch['lengths'])
            
            # Compute loss and accuracy
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
            epoch_loss += loss
            
            predictions = np.argmax(logits, axis=-1)
            correct += np.sum(predictions == labels)
            total += len(labels)
        
        # Validation
        val_correct = 0
        val_total = 0
        val_loss = 0
        confusion = np.zeros((model.num_classes, model.num_classes))
        
        for i in range(0, len(val_data), batch_size):
            batch_data = val_data[i:i+batch_size]
            texts = [text for text, _ in batch_data]
            labels = np.array([label for _, label in batch_data])
            
            batch = text_processor.create_batches(texts, batch_size)[0]
            logits = model.forward(batch['input_ids'], batch['lengths'])
            
            # Compute metrics
            predictions = np.argmax(logits, axis=-1)
            val_correct += np.sum(predictions == labels)
            val_total += len(labels)
            
            # Update confusion matrix
            for pred, true in zip(predictions, labels):
                confusion[true, pred] += 1
        
        # Update results
        results['train_loss'].append(epoch_loss / (len(train_data) / batch_size))
        results['train_accuracy'].append(correct / total)
        results['val_accuracy'].append(val_correct / val_total)
        results['confusion_matrix'] = confusion
    
    return results

def update_model_weights(model, learning_rate=0.01):
    """
    Simple SGD weight update for any model.
    This applies the accumulated gradients to update weights.
    """
    if hasattr(model, 'cell'):
        cell = model.cell
        
        # Update based on cell type
        if hasattr(cell, 'W_xf'):  # LSTM
            cell.W_xf -= learning_rate * np.clip(cell.dW_xf, -5, 5)
            cell.W_hf -= learning_rate * np.clip(cell.dW_hf, -5, 5)
            cell.b_f -= learning_rate * np.clip(cell.db_f, -5, 5)
            
            cell.W_xi -= learning_rate * np.clip(cell.dW_xi, -5, 5)
            cell.W_hi -= learning_rate * np.clip(cell.dW_hi, -5, 5)
            cell.b_i -= learning_rate * np.clip(cell.db_i, -5, 5)
            
            cell.W_xc -= learning_rate * np.clip(cell.dW_xc, -5, 5)
            cell.W_hc -= learning_rate * np.clip(cell.dW_hc, -5, 5)
            cell.b_c -= learning_rate * np.clip(cell.db_c, -5, 5)
            
            cell.W_xo -= learning_rate * np.clip(cell.dW_xo, -5, 5)
            cell.W_ho -= learning_rate * np.clip(cell.dW_ho, -5, 5)
            cell.b_o -= learning_rate * np.clip(cell.db_o, -5, 5)
            
            # Reset gradients
            cell.dW_xf.fill(0); cell.dW_hf.fill(0); cell.db_f.fill(0)
            cell.dW_xi.fill(0); cell.dW_hi.fill(0); cell.db_i.fill(0)
            cell.dW_xc.fill(0); cell.dW_hc.fill(0); cell.db_c.fill(0)
            cell.dW_xo.fill(0); cell.dW_ho.fill(0); cell.db_o.fill(0)
            
        elif hasattr(cell, 'W_xr'):  # GRU
            cell.W_xr -= learning_rate * np.clip(cell.dW_xr, -5, 5)
            cell.W_hr -= learning_rate * np.clip(cell.dW_hr, -5, 5)
            cell.b_r -= learning_rate * np.clip(cell.db_r, -5, 5)
            
            cell.W_xz -= learning_rate * np.clip(cell.dW_xz, -5, 5)
            cell.W_hz -= learning_rate * np.clip(cell.dW_hz, -5, 5)
            cell.b_z -= learning_rate * np.clip(cell.db_z, -5, 5)
            
            cell.W_xh -= learning_rate * np.clip(cell.dW_xh, -5, 5)
            cell.W_hh -= learning_rate * np.clip(cell.dW_hh, -5, 5)
            cell.b_h -= learning_rate * np.clip(cell.db_h, -5, 5)
            
            # Reset gradients
            cell.dW_xr.fill(0); cell.dW_hr.fill(0); cell.db_r.fill(0)
            cell.dW_xz.fill(0); cell.dW_hz.fill(0); cell.db_z.fill(0)
            cell.dW_xh.fill(0); cell.dW_hh.fill(0); cell.db_h.fill(0)
            
        elif hasattr(cell, 'W_xh'):  # Vanilla RNN
            cell.W_xh -= learning_rate * np.clip(cell.dW_xh, -5, 5)
            cell.W_hh -= learning_rate * np.clip(cell.dW_hh, -5, 5)
            cell.b_h -= learning_rate * np.clip(cell.db_h, -5, 5)
            
            # Reset gradients
            cell.dW_xh.fill(0)
            cell.dW_hh.fill(0)
            cell.db_h.fill(0)

def load_shakespeare_text(filepath='shakespeare.txt'):
    """Load Shakespeare text data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def load_movie_reviews(filepath='reviews.csv'):
    """Load movie reviews data."""
    import csv
    reviews = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                reviews.append(row[0])
                labels.append(int(row[1]))
    return list(zip(reviews, labels))


def main():
    """Main training function with proper weight updates."""

    import os

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    print("="*70)
    print("CS5720 - Assignment 3: RNNs and Transformers")
    print("="*70)
    
    # PART 1: Text Generation
    print("\n" + "="*70)
    print("PART 1: Text Generation with Shakespeare")
    print("="*70)
    
    text = load_shakespeare_text()
    print(f"Loaded {len(text)} characters")
    
    # Create smaller chunks for better learning
    chunk_size = 50
    chunks = [text[i:i+chunk_size] for i in range(0, len(text)-chunk_size, 25)]
    
    split = int(0.8 * len(chunks))
    train_chunks = chunks[:split]
    
    print(f"Training chunks: {len(train_chunks)}")
    
    # Initialize processor
    processor = TextProcessor(level='char')
    processor.build_vocab(train_chunks)
    print(f"Vocabulary size: {processor.vocab_size}")
    
    # Train LSTM language model
    print("\nTraining LSTM Language Model...")
    model = RNNLanguageModel(processor.vocab_size, hidden_dim=64, cell_type='lstm')
    
    num_epochs = 15
    learning_rate = 0.01  
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_updates = 0
        
        # Processing each chunk individually for simpler training
        for chunk_text in train_chunks[:50]:  # Using first 50 chunks
            # Encode chunk
            encoded = processor.encode(chunk_text)
            if len(encoded) < 2:
                continue
                
            # Create simple batch
            input_ids = encoded[:-1][np.newaxis, :]
            target_ids = encoded[1:]
            
            # Forward pass
            h = np.zeros((1, model.hidden_dim))
            c = np.zeros((1, model.hidden_dim))
            
            chunk_loss = 0
            # Process each timestep
            for t in range(len(input_ids[0])):
                x_t = model.embedding[input_ids[0, t]][np.newaxis, :]
                
                # Forward through LSTM
                h, c, cache = model.cell.forward(x_t, h, c)
                
                # Compute logits
                logits = np.dot(h, model.output_proj) + model.b_proj
                
                # Softmax
                scores = logits - np.max(logits)
                probs = np.exp(scores) / np.sum(np.exp(scores))
                
                # Loss
                target = target_ids[t]
                if target < len(probs[0]):
                    chunk_loss -= np.log(probs[0, target] + 1e-10)
                    
                    # Backward pass (simplified)
                    dlogits = probs.copy()
                    dlogits[0, target] -= 1
                    
                    # Update output weights
                    model.output_proj -= learning_rate * np.dot(h.T, dlogits)
                    model.b_proj -= learning_rate * dlogits[0]
                    
                    # Backprop through LSTM (one step)
                    dh = np.dot(dlogits, model.output_proj.T)
                    dx, dh_prev, dc_prev = model.cell.backward(dh, np.zeros_like(h), cache)
                    
                    # Update embedding
                    model.embedding[input_ids[0, t]] -= learning_rate * dx[0]
            
            # Update LSTM weights after processing chunk
            update_model_weights(model, learning_rate)
            
            epoch_loss += chunk_loss / max(len(input_ids[0]), 1)
            num_updates += 1
        
        avg_loss = epoch_loss / max(num_updates, 1)
        perplexity = np.exp(min(avg_loss, 10))  # Cap for stability
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        
        # Generate sample
        if (epoch + 1) % 5 == 0:
            prompt = "To be"
            prompt_ids = processor.encode(prompt)[None, :]
            if len(prompt_ids[0]) > 0:
                generated = model.generate(prompt_ids, max_length=80, temperature=0.7)
                generated_text = processor.decode(generated[0])
                print(f"Sample: {generated_text[:100]}...")
    
    # PART 2: Sentiment Analysis
    print("\n" + "="*70)
    print("PART 2: Sentiment Analysis with Movie Reviews")
    print("="*70)
    
    data = load_movie_reviews()
    print(f"Loaded {len(data)} reviews")
    
    # Split
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    print(f"Training: {len(train_data)}, Validation: {len(val_data)}")
    
    # Build vocabulary
    processor_sent = TextProcessor(level='word')
    texts = [text for text, _ in train_data]
    processor_sent.build_vocab(texts, min_freq=1)
    print(f"Vocabulary size: {processor_sent.vocab_size}")
    
    # Initialize classifier with better parameters
    model_clf = BiLSTMClassifier(
        vocab_size=processor_sent.vocab_size,
        hidden_dim=64,  
        num_classes=2,
        use_attention=True  # Enable attention for better context understanding
    )
    
    print("\nTraining BiLSTM Classifier...")
    num_epochs = 30  
    batch_size = 8 
    learning_rate = 0.01  
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = [train_data[i] for i in indices]
        
        # Training
        correct = 0
        total = 0
        
        # Process in mini-batches
        for i in range(0, len(train_data_shuffled), batch_size):
            batch_data = train_data_shuffled[i:i + batch_size]
            batch_texts = [text for text, _ in batch_data]
            batch_labels = np.array([label for _, label in batch_data])
            
            # Prepare batch
            max_len = max(len(processor_sent.encode(text)) for text in batch_texts)
            batch_input_ids = np.zeros((len(batch_data), max_len), dtype=np.int32)
            batch_lengths = np.zeros(len(batch_data), dtype=np.int32)
            
            for j, text in enumerate(batch_texts):
                encoded = processor_sent.encode(text)
                batch_input_ids[j, :len(encoded)] = encoded
                batch_lengths[j] = len(encoded)
            
            # Forward pass
            logits = model_clf.forward(batch_input_ids, batch_lengths)
            
            # Compute probabilities with improved numerical stability
            scores = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            
            # Predictions
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == batch_labels)
            total += len(batch_labels)
            
            # Gradient computation
            dlogits = probs.copy()
            dlogits[np.arange(len(batch_labels)), batch_labels] -= 1
            dlogits /= len(batch_labels)  # Average over batch
            
            # Use the stored hidden states
            final_states = model_clf.hidden_states
            
            # Gradient clipping
            grad_norm = np.linalg.norm(dlogits)
            if grad_norm > 1.0:
                dlogits = dlogits / grad_norm
            
            # Update classifier weights and biases
            final_state_mean = np.mean(final_states, axis=1)  # Average over sequence length
            model_clf.classifier -= learning_rate * np.dot(final_state_mean.T, dlogits)
            model_clf.b_classifier -= learning_rate * np.sum(dlogits, axis=0)
        
        train_acc = correct / max(total, 1)
        
        # Validation
        val_correct = 0
        val_total = 0
        
        for text, label in val_data:
            encoded = processor_sent.encode(text)
            if len(encoded) == 0:
                continue
            
            input_ids = encoded[np.newaxis, :]
            lengths = np.array([len(encoded)])
            
            logits = model_clf.forward(input_ids, lengths)
            pred = np.argmax(logits)
            
            if pred == label:
                val_correct += 1
            val_total += 1
        
        val_acc = val_correct / max(val_total, 1)
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if (epoch + 1) % 10 == 0:
                print(f"  → New best: {best_val_acc:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Final Best Validation Accuracy: {best_val_acc:.4f}")
    
    print("="*70)
    print("Training Completed!")
    print("="*70)

if __name__ == "__main__":
    main()