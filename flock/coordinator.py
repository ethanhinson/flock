"""
flock.coordinator — the machine that starts the lap.

Holds the embedding, the first block of layers, and the vocab projection. Every
token's hidden state leaves here, travels the chain of birds, and comes back to
be turned into a token id.
"""
import torch


class LayerCache:
    """Per-layer K/V for the coordinator's own layers.

    This is HALF a cache: each bird independently keeps K/V for its own layers.
    Conversation state is sharded exactly like the weights are -- no single
    device holds all of it.
    """
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.k is not None:
            k = torch.cat([self.k, k], dim=2)
            v = torch.cat([self.v, v], dim=2)
        self.k, self.v = k, v
        return k, v


class Coordinator:
    def __init__(self, model, cut):
        self.model = model
        self.cut = cut
        model.model.layers = torch.nn.ModuleList(model.model.layers[:cut])
        self.embed = model.model.embed_tokens
        self.rotary = model.model.rotary_emb
        self.lm_head = model.lm_head
        self.layers = model.model.layers

    def new_caches(self):
        return [LayerCache() for _ in self.layers]

    def forward(self, ids, caches=None, offset=0):
        """Embedding + our layers -> hidden state to hand to the first bird.

        With a cache, `ids` is just the NEW tokens (usually one) and `offset`
        says where they sit, so this is O(1) per step instead of re-running the
        whole prompt every time.
        """
        q = len(ids)
        pids = torch.arange(offset, offset + q)[None, :]
        h = self.embed(torch.tensor([ids]))
        pos = self.rotary(h, pids)
        kv = offset + q
        m = torch.full((q, kv), torch.finfo(torch.float32).min)
        m = torch.triu(m, diagonal=1 + (kv - q))[None, None]
        with torch.no_grad():
            for i, lyr in enumerate(self.layers):
                if caches is not None:
                    out = lyr(h, attention_mask=m, position_ids=pids,
                              past_key_values=caches[i], use_cache=True,
                              position_embeddings=pos)
                else:
                    out = lyr(h, attention_mask=m, position_ids=pids,
                              position_embeddings=pos)
                h = out[0] if isinstance(out, tuple) else out
        return h

    def project(self, h_last):
        """Tied vocab projection -- kept here so the birds stay small."""
        with torch.no_grad():
            return int(self.lm_head(h_last).argmax(-1).item())
