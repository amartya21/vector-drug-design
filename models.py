import torch
import torch.nn.functional as F
import numpy as np

class ContrastiveSMILESModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(ContrastiveSMILESModel, self).__init__()
        self.bert = bert_model
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ELU(),
            torch.nn.Linear(768, 768)
        )
        device = next(bert_model.parameters()).device
        self.projection = self.projection.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(embeddings)
        return embeddings, projected

    def get_embedding(self, smiles, tokenizer, device=None):
        if device is None:
            device = next(self.bert.parameters()).device
        tokens = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings, _ = self.forward(tokens.input_ids, tokens.attention_mask)
            emb = embeddings.cpu().squeeze().numpy()
            if isinstance(emb, np.ndarray) and emb.ndim > 1 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
        return emb

class SMILESDecoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bert_embed_dim=768, max_length=150, n_layers=3, dropout=0.2, device='cuda'):
        super(SMILESDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.n_layers = n_layers
        self.device = device
        self.linear_init = torch.nn.Sequential(
            torch.nn.Linear(bert_embed_dim, hidden_dim * n_layers),
            torch.nn.LayerNorm(hidden_dim * n_layers),
            torch.nn.ELU()
        )
        self.char_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.fc_out = torch.nn.Linear(hidden_dim, vocab_size)
        self.to(device)

    def forward(self, bert_embeddings, target_tensors=None):
        batch_size = bert_embeddings.size(0)
        hidden = self.linear_init(bert_embeddings)
        hidden = hidden.view(batch_size, self.n_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        if target_tensors is None:
            return None
        inp = target_tensors[:, :-1]
        tgt = target_tensors[:, 1:]
        inp_emb = self.embed_dropout(self.char_embedding(inp))
        outputs, _ = self.gru(inp_emb, hidden)
        projected = self.projection(outputs)
        logits = self.fc_out(projected)
        logits_flat = logits.reshape(-1, self.vocab_size)
        tgt_flat = tgt.reshape(-1)
        loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=0)
        return loss

    def generate(self, bert_embedding, idx2char, max_length=150, temperature=1.0, device=None):
        if device is None:
            device = self.device
        if bert_embedding.dim() > 2:
            bert_embedding = bert_embedding.squeeze()
        if bert_embedding.dim() == 1:
            bert_embedding = bert_embedding.unsqueeze(0)
        out = self.generate_batch(bert_embedding, idx2char, max_length=max_length, temperature=temperature, device=device)
        return out[0]

    def generate_batch(self, bert_embeddings, idx2char, max_length=150, temperature=1.0, device=None):
        if device is None:
            device = self.device
        self.eval()
        bert_embeddings = bert_embeddings.to(device)
        bsz = bert_embeddings.size(0)
        hidden = self.linear_init(bert_embeddings)
        hidden = hidden.view(bsz, self.n_layers, self.hidden_dim).transpose(0, 1).contiguous()
        cur = torch.full((bsz, 1), 1, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        tokens = []
        for _ in range(max_length):
            emb = self.char_embedding(cur)
            out, hidden = self.gru(emb, hidden)
            proj = self.projection(out)
            logits = self.fc_out(proj).squeeze(1) / max(1e-6, temperature)
            next_tok = torch.distributions.Categorical(logits=logits).sample()
            tokens.append(next_tok)
            finished = finished | (next_tok == 2)
            cur = next_tok.unsqueeze(1)
            if finished.all():
                break
        toks = torch.stack(tokens, dim=1).detach().cpu().numpy()
        smiles = []
        for row in toks:
            chars = []
            for t in row:
                if int(t) == 2:
                    break
                chars.append(idx2char.get(int(t), ''))
            smiles.append(''.join(chars))
        return smiles
