class RotationLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.rotation_matrix = nn.Parameter(torch.eye(embed_dim))

    def forward(self, x):
        # Apply the rotational transformation
        rotated_x = torch.matmul(x, self.rotation_matrix)
        return rotated_x

    def reset_parameters(self):
        nn.init.orthogonal_(self.rotation_matrix)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def rotate_queries_or_keys(self, x):
        sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(x.shape[1], device=x.device), self.inv_freq) 
        sin = sinusoid_inp.sin()[None, :, None, :] 
        cos = sinusoid_inp.cos()[None, :, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x

##### combined into one class

class RotaryEmbeddingWithRotation(nn.Module):
    def __init__(self, n_state, n_head, base=10000):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        
        self.rotation_matrix = nn.Parameter(torch.eye(self.h_dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

    def reset_parameters(self):
        nn.init.orthogonal_(self.rotation_matrix)

    def forward(self, x):
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

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq)
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        x1, x2 = rotated_x[..., ::2], rotated_x[..., 1::2]
        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        rotated_x = rotated_x.reshape(batch_size, seq_len, self.n_state)
        return rotated_x

###### integrated into multihead
class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, base: int = 10000, checkpointing=False):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.h_dim = n_state // n_head
        self.checkpointing=checkpointing

        self.rotation_matrix = nn.Parameter(torch.empty(self.h_dim, self.h_dim))
        nn.init.orthogonal_(self.rotation_matrix)
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)

    def rotate_queries_or_keys(self, x):
        sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(x.shape[1], device=x.device), self.inv_freq)
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x

    def forward(self, x: torch.Tensor, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1)

        q = self.rotate_queries_or_keys(q)
        k = self.rotate_queries_or_keys(k)
        q = torch.matmul(q, self.rotation_matrix)
        k = torch.matmul(k, self.rotation_matrix)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk

###### encoder


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, checkpointing=False):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.rotor_layer = RotaryEmbeddingWithRotation(n_state, n_head)
        self.checkpointing = checkpointing

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, checkpointing=checkpointing) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        x = x.permute(0, 2, 1)

        x = self.rotor_layer(x)
        
        pos_emb = self.positional_embedding(torch.arange(x.size(1), device=x.device)).unsqueeze(0)
        x = x + pos_emb

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_post(x)
        return x

####### Decoder

class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.rotor_layer = RotaryEmbeddingWithRotation(n_state, n_head)
        self.checkpointing = checkpointing
        self.n_head = n_head

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=True, checkpointing=checkpointing)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)

        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)

        batch_size, seq_length, embedding_dim = x.shape
        num_heads = self.n_head
        head_dim = embedding_dim // num_heads
        x = x.view(batch_size, seq_length, num_heads, head_dim)

        x = self.rotor_layer(x)
        x = x.view(batch_size, seq_length, embedding_dim)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x, xa, self.mask, kv_cache)
            else:
                x = block(x, xa, self.mask, kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

