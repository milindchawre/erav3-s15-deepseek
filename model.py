import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_weights(module, std=0.041666666666666664):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(self.max_position_embeddings).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            seq_len = self.max_position_embeddings
            
        return (
            self.cos_cached[:,:,:seq_len,:],
            self.sin_cached[:,:,:seq_len,:]
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Ensure proper broadcasting
    cos = cos[:, :, :q.size(2), :]  # [batch, 1, seq_len, dim]
    sin = sin[:, :, :q.size(2), :]  # [batch, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LatentAttention(nn.Module):
    def __init__(self, config, compression_ratio=8):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.latent_size = self.hidden_size // compression_ratio
        
        # Latent projections
        self.to_latent = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.from_latent = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.to_latent.weight, std=0.02)
        nn.init.normal_(self.from_latent.weight, std=0.02)

    def forward(self, x):
        # Project to latent space
        latent = self.to_latent(x)
        # Project back to original space
        return self.from_latent(latent)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Add latent attention
        self.latent_attn = LatentAttention(config)
        self.latent_gate = nn.Parameter(torch.ones(1))
        
        self.kv_cache = None
    
    def forward(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Regular attention path
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat k/v heads if needed
        if self.num_key_value_heads != self.num_attention_heads:
            k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        
        # Compute attention
        attn_output = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_output = attn_output + attention_mask
            
        attn_output = F.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_output, v)
        
        # Process regular attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        
        # Latent attention path
        latent_output = self.latent_attn(hidden_states)
        
        # Combine outputs using learned gate
        output = attn_output + self.latent_gate * latent_output
        
        return output

class ExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = 8
        self.num_shared_experts = 1
        self.top_k = 2
        self.hidden_size = config["hidden_size"]
        
        # Create experts
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_experts)])
        self.shared_experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_shared_experts)])
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts + self.num_shared_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Get router logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts + num_shared]
        
        # Calculate routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Reshape input for expert computation
        x_reshaped = x.view(-1, hidden_size)
        
        # Initialize output tensor
        combined_output = torch.zeros_like(x_reshaped)
        
        # Process through experts
        for i in range(self.top_k):
            expert_indices = top_k_indices[..., i].view(-1)
            expert_weights = top_k_weights[..., i].view(-1, 1)
            
            # Process each expert
            for expert_idx in range(self.num_experts + self.num_shared_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    if expert_idx < self.num_experts:
                        expert_output = self.experts[expert_idx](x_reshaped[mask])
                    else:
                        expert_output = self.shared_experts[expert_idx - self.num_experts](x_reshaped[mask])
                    combined_output[mask] += expert_output * expert_weights[mask]
        
        return combined_output.view(batch_size, seq_len, hidden_size)

# Replace the existing MLP class with MoEMLP in DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MoEMLP(config)  # Changed from MLP to MoEMLP
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        
    def forward(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, use_cache)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# Rename SmolLM2 class to DeepSeek
class DeepSeek(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(
            config["hidden_size"] // config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            theta=config.get("rope_theta", 10000.0)
        )
        
        # Enable gradient checkpointing
        self.gradient_checkpointing = True
        
        # Initialize weights
        self.apply(lambda p: _init_weights(p, std=config.get("initializer_range", 0.041666666666666664)))
        
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        
        seq_length = input_ids.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_length)
        
        # Apply gradient checkpointing to reduce memory usage
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, cos, sin, attention_mask, use_cache
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, cos, sin, attention_mask, use_cache)
        
        hidden_states = self.norm(hidden_states)
        
        # Use tied weights for the output projection
        if self.config.get("tie_word_embeddings", True):
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        return logits

    def generate(
        self, 
        input_ids, 
        max_length, 
        min_length=None,
        num_return_sequences=1, 
        pad_token_id=None,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    ):
        self.eval()
        batch_size = input_ids.shape[0]
        min_length = min_length if min_length is not None else input_ids.shape[1]
        
        # Clear KV cache
        for layer in self.layers:
            layer.self_attn.kv_cache = None
            
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids, use_cache=True)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Stop if all sequences have hit the pad token
                if pad_token_id is not None and (next_tokens == pad_token_id).all():
                    break
                
                # Stop if we've reached min_length
                if input_ids.shape[1] < min_length:
                    continue
                    
        return input_ids