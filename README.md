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


class MultiHeadAttention(nn.Module):
    sdpa = True

    def __init__(self, n_state: int, n_head: int, dropout_rate=0.01, gradient_checkpointing=False, group_norm=False):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        self.rotation_layer = RotationLayer(self.head_dim) 
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))
        self.dropout = nn.Dropout(dropout_rate)

        self.group_norm = group_norm
        self.attn_ln = GroupNorm(num_groups=1, num_channels=n_state) if group_norm else nn.LayerNorm(n_state)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        x_norm = self.attn_ln(x)

        q = self.query(x_norm)
        k = self.key(x_norm if xa is None else xa)
        v = self.value(x_norm if xa is None else xa)

        if kv_cache is not None and self.key in kv_cache:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1)

        # Apply rotary embeddings and rotational layer
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        q = self.rotation_layer(q)
        k = self.rotation_layer(k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        wv, qk = self.qkv_attention(q, k, v, mask)

        return self.out(wv) + x, qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = self.temperature

        q = q.view(n_batch, n_ctx, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(n_batch, k.shape[1], self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(n_batch, v.shape[1], self.n_head, self.head_dim).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
            qk = None
        else:
            qk = (q * scale) @ (k.transpose(-2, -1) * scale)
            if mask is not None:
                qk += mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            w = self.dropout(w)
            out = (w @ v).permute(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
            qk = qk.detach()

        return out, qk
