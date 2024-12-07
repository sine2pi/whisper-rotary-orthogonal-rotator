import torch, math, warnings, os, gzip, base64, transformers, numpy as np
import torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
from transformers import WhisperPreTrainedModel, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from huggingface_hub import PyTorchModelHubMixin

from whisper.decoding import decode as decode_function
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate = 0.01, use_batchnorm: bool = True, activation: str = 'relu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity=self.activation)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, x.size(-1))  
        x = self.linear(x)

        if self.use_batchnorm:
            x = self.batchnorm(x)

        x = self.apply_activation(x)
        x = self.dropout(x)
        x = x.view(batch_size, seq_len, -1)  
        
        return x

    def apply_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}')

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _conv_forward(self, x, weight, bias) -> Tensor:
        weight = self.weight.to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return super()._conv_forward(x, weight, bias)

def givens_rotation_matrix(n_state, i, j, theta):
    G = torch.eye(n_state)
    G[i, i] = math.cos(theta)
    G[i, j] = -math.sin(theta)
    G[j, i] = math.sin(theta)
    G[j, j] = math.cos(theta)
    return G

class GivensRotations(nn.Module):
    def __init__(self, h_dim, num_rotations):
        super().__init__()
        self.h_dim = h_dim
        self.num_rotations = num_rotations
        self.thetas = nn.Parameter(torch.zeros(num_rotations))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor to be 4D, but got {x.dim()}D")
        
        batch_size, seq_len, n_head, h_dim = x.size()
        
        if h_dim != self.h_dim:
            raise ValueError(f"Expected h_dim of {self.h_dim}, but got {h_dim}")
        
        x = x.view(-1, h_dim) 
        for k in range(self.num_rotations):
            i, j = k % self.h_dim, (k + 1) % self.h_dim
            G = givens_rotation_matrix(self.h_dim, i, j, self.thetas[k])
            x = torch.matmul(x, G.to(x.device))
        
        x = x.view(batch_size, seq_len, n_head, h_dim)  
        return x

class BiasedCrossAttention(nn.Module):
    def __init__(self, n_state, n_head, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.bias = nn.Parameter(torch.zeros(n_head, 1, self.head_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNorm(n_state)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()

        q = self.query(q).view(batch_size, seq_length, self.n_head, self.head_dim)
        k = self.key(k).view(batch_size, seq_length, self.n_head, self.head_dim)
        v = self.value(v).view(batch_size, seq_length, self.n_head, self.head_dim)

        qk = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) + self.bias
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))

        w = F.softmax(qk, dim=-1)
        w = self.dropout(w)

        out = (w @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        out = self.norm(self.out(out) + q.view(batch_size, seq_length, -1))
        return out

class DynamicConvAttention(nn.Module):
    def __init__(self, n_state, n_head, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(n_state, n_state, kernel_size, padding=kernel_size // 2, groups=n_head)
        self.dropout = nn.Dropout(dropout_rate)

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out_proj = nn.Linear(n_state, n_state)

        self.norm = LayerNorm(n_state)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim != self.n_state:
            raise ValueError(f"Expected embed_dim of {self.n_state}, but got {embed_dim}")

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        x = x.permute(0, 2, 1)
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.norm(conv_out)
        conv_out = self.dropout(conv_out)

        attention_out = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.n_state ** 0.5), dim=-1)
        attention_out = torch.matmul(attention_out, v)
        
        combined_out = conv_out + attention_out
        combined_out = self.norm(combined_out)
        
        return self.out_proj(self.dropout(combined_out)) + x.permute(0, 2, 1)

class HybridAttention(nn.Module):
    def __init__(self, n_state, n_head, window_size=1, dropout_rate=0.1):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(n_state, n_head, dropout=dropout_rate)
        self.global_attn = nn.MultiheadAttention(n_state, n_head, dropout=dropout_rate)
        self.ln_local = LayerNorm(n_state)
        self.ln_global = LayerNorm(n_state)

        self.dropout = nn.Dropout(dropout_rate)
        self.window_size = window_size

    def forward(self, x):
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)
        x_global = x_global.permute(1, 0, 2)
        local_out = self.sliding_window_attention(x_local)
        global_out, _ = self.global_attn(x_global, x_global, x_global)
        combined_out = local_out + global_out
        combined_out = combined_out.permute(1, 0, 2)
        return self.dropout(combined_out)

    def sliding_window_attention(self, x):
        seq_len, batch_size, n_state = x.size()
        window_size = min(self.window_size, max(1, seq_len // 4))
        output = torch.zeros_like(x, device=x.device, dtype=x.dtype)

        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - window_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]

        return output
    
class RotaryEmbeddingWithRotation(nn.Module):
    def __init__(self, n_state, n_head, base=10000, checkpointing=False):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.base = base  # Initialize base
        self.checkpointing = checkpointing

        self.rotation_matrix = nn.Parameter(torch.eye(self.h_dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

    def update_base(self, new_base):
        self.base = new_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

    def reset_parameters(self):
        nn.init.orthogonal_(self.rotation_matrix)

    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, n_state = x.size()
        elif x.dim() == 4:
            batch_size, seq_len, n_head, h_dim = x.size()
            n_state = n_head * h_dim
            x = x.view(batch_size, seq_len, n_state)
        else:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        if n_state != self.n_state:
            raise ValueError(f"Expected n_state of {self.n_state}, but got {n_state}")

        x = x.reshape(batch_size, seq_len, self.n_head, self.h_dim)
        x = x.reshape(-1, self.h_dim)
        rotated_x = torch.matmul(x, self.rotation_matrix)
        rotated_x = rotated_x.reshape(batch_size, seq_len, self.n_head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq.to(x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        x1, x2 = rotated_x[..., ::2], rotated_x[..., 1::2]
        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        rotated_x = rotated_x.reshape(batch_size, seq_len, self.n_state)
        return rotated_x

class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx, n_state, checkpointing=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.checkpointing = checkpointing

        position = torch.arange(0, n_ctx, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_state, 2).float() * -(math.log(10000.0) / n_state))
        features = torch.zeros(n_ctx, n_state)
        features[:, 0::2] = torch.sin(position * div_term)
        features[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_features', features)

        self.positional_embeddings = nn.Parameter(self.sinusoidal_features.clone())

    def forward(self, positions):
        if self.checkpointing:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]

        position_embeddings = torch.nn.functional.normalize(position_embeddings, p=2, dim=-1)
        return position_embeddings

class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, base: int = 10000, max_rel_dist: int = 1):
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.n_head = n_head
        self.h_dim = n_state // n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.positional_scaling = nn.Parameter(torch.ones(1))

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.max_rel_dist = max_rel_dist
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

        self.rotary_embedding = RotaryEmbeddingWithRotation(n_state, n_head, base=10000)

        self.rotation_matrix = nn.Parameter(torch.empty(self.h_dim, self.h_dim))
        nn.init.orthogonal_(self.rotation_matrix)

        self.givens_rotations = GivensRotations(self.h_dim, num_rotations=self.h_dim // 2) 

        self.rel_pos_bias = nn.Embedding(2 * self.max_rel_dist - 1, self.n_head)
        self.rel_pos_bias.weight.data.fill_(0)

        if device:
            self.to(device)

    def update_base(self, new_base): 
        self.base = new_base 
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.rotary_embedding.update_base(new_base)

    def apply_rotary_embedding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        scaled_positions = self.positional_scaling * positions
        sinusoid_inp = torch.outer(scaled_positions, self.inv_freq.to(x.device)) 
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated

    def forward(self, x, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or 'k' not in kv_cache:
            k_input = x if xa is None else xa
            k = self.key(k_input)
            v = self.value(k_input)
            if kv_cache is not None:
                kv_cache['k'] = k
                kv_cache['v'] = v
        else:
            k = kv_cache['k']
            v = kv_cache['v']

        q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1)
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1)

        q = self.apply_rotary_embedding(q)
        k = self.apply_rotary_embedding(k)

        q = torch.matmul(q, self.rotation_matrix)
        k = torch.matmul(k, self.rotation_matrix)

        # q = self.givens_rotations(q) 
        # k = self.givens_rotations(k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
    
    def qkv_attention(self, q, k, v, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = torch.arange(seq_len_q, device=q.device).unsqueeze(1) - torch.arange(seq_len_k, device=q.device).unsqueeze(0)
        positions = positions.clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias(positions)  
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  

        qk = qk + rel_bias

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()

        return out, qk
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, max_rel_dist = 1, checkpointing=False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.checkpointing = checkpointing
        self.max_rel_dist = max_rel_dist

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, kv_cache: Optional[dict] = None):
        if self.checkpointing:
            x = checkpoint(self._attn_forward, x, mask, kv_cache)
        else:
            x = self._attn_forward(x, mask, kv_cache)

        if self.cross_attn:
            if self.checkpointing:
                x = checkpoint(self._cross_attn_forward, x, xa, kv_cache)
            else:
                x = self._cross_attn_forward(x, xa, kv_cache)

        if self.checkpointing:
            x = checkpoint(self._mlp_forward, x)
        else:
            x = self._mlp_forward(x)

        return x

    def _attn_forward(self, x, mask, kv_cache):
        residual = x
        x = self.attn_ln(x)
        x = residual + self.attn(x, mask=mask, kv_cache=kv_cache)[0]
        return x

    def _cross_attn_forward(self, x, xa, kv_cache):
        residual = x
        x = self.cross_attn_ln(x)
        x = residual + self.cross_attn(x, xa, kv_cache=kv_cache)[0]
        return x

    def _mlp_forward(self, x):
        residual = x
        x = self.mlp_ln(x)
        x = residual + self.mlp(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_rel_dist, checkpointing=False):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.rotary_embedding = RotaryEmbeddingWithRotation(n_state, n_head, base=10000)
        self.checkpointing = checkpointing

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, max_rel_dist, checkpointing=checkpointing) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def update_base(self, new_base):
        self.rotary_embedding.update_base(new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention):
                block.cross_attn.update_base(new_base)

    def forward(self, x):
        if self.checkpointing:
            x = checkpoint(self._conv_forward, x)
        else:
            x = self._conv_forward(x)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_post(x)
        return x

    def _conv_forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.rotary_embedding(x)
        
        pos_emb = self.positional_embedding(torch.arange(x.size(1), device=x.device)).unsqueeze(0)
        x = x + pos_emb
        return x

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_state, n_head, n_layer, max_rel_dist, cross_attention, checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.rotary_embedding = RotaryEmbeddingWithRotation(n_state, n_head, base=10000)
        self.checkpointing = checkpointing
        self.n_head = n_head

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, max_rel_dist, cross_attention, checkpointing=checkpointing)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def update_base(self, new_base):
        self.rotary_embedding.update_base(new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention):
                block.cross_attn.update_base(new_base)

    def forward(self, x, xa, kv_cache: Optional[dict] = None):
        if self.checkpointing:
            x = checkpoint(self._embedding_forward, x, xa, kv_cache)
        else:
            x = self._embedding_forward(x, xa, kv_cache)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x, xa, self.mask, kv_cache)
            else:
                x = block(x, xa, self.mask, kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)

        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)

        batch_size, seq_length, embedding_dim = x.shape
        num_heads = self.n_head
        head_dim = embedding_dim // num_heads
        x = x.view(batch_size, seq_length, num_heads, head_dim)

        x = self.rotary_embedding(x)
        x = x.view(batch_size, seq_length, embedding_dim)
        return x

class mymodel(WhisperPreTrainedModel, PyTorchModelHubMixin):
    config_class = WhisperConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.config = config
        
        self.encoder = AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            self.config.checkpointing,
            self.config.max_rel_dist
        )
        self.decoder = TextDecoder(
            self.config.vocab_size,
            self.config.n_text_ctx,
            self.config.n_text_state,
            self.config.n_text_head,
            self.config.n_text_layer,
            self.config.checkpointing,
            self.config.max_rel_dist
        )

        all_heads = torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool)
        all_heads[self.config.n_text_layer // 2:] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.best_loss = float('inf')
        self.base = 10000 

    def update_base(self, new_base):
        self.encoder.rotary_embedding.update_base(new_base)
        self.decoder.rotary_embedding.update_base(new_base)
        for name, module in self.encoder.named_modules():
            if isinstance(module, MultiHeadAttention):
                module.update_base(new_base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, MultiHeadAttention):
                module.update_base(new_base)

    def adjust_base(self, loss, factor=1.05):
        if loss < self.best_loss:
            new_base = self.base * factor
        else:
            new_base = self.base / factor

        self.update_base(new_base)
        self.best_loss = loss
        #print(f"Adjusted base: {new_base}")


    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id) -> torch.Tensor:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None):
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(device)
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) 
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            self.adjust_base(loss.item())

        return {
            "loss": loss,
            "logits": logits,
            "input_features": encoded_features,
            "labels": labels,
            "decoder_input_ids": dec_input_ids
        }

    def apply_initialization(self):
        self._initialize_weights()

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.config.n_text_layer, self.config.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, labels, input_features):
        return self.decoder(labels, input_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.config.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.config.vocab_size - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.config.n_text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    def set_input_embeddings(self, new_embeddings: torch.nn.Embedding):
        self.decoder.token_embedding = new_embeddings

    def get_input_embeddings(self):
        return self.decoder.token_embedding

    def resize_token_embeddings(self, new_num_tokens: int):
        old_embeddings = self.get_input_embeddings()
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = torch.nn.Embedding(new_num_tokens, old_embedding_dim)
    
        new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data
        self.set_input_embeddings(new_embeddings)
        self.config.vocab_size = new_num_tokens

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

    def get_encoder(self):
        return self.encoder

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_features': input_ids}

    def _prepare_decoder_input_ids_for_generation(self, batch_size, decoder_start_token_id=None, bos_token_id=None):
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.config.decoder_start_token_id

    def can_generate(self):
        return True
    
    def generate(self, inputs, **kwargs):
        encoder_outputs = self.encoder(inputs)
        decoder_input_ids = torch.zeros((inputs.size(0), 1), dtype=torch.long, device=inputs.device)
        outputs = self.decoder(decoder_input_ids, encoder_outputs)
        return outputs.argmax(dim=-1)

#rasa
