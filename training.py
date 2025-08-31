import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import smiles_to_tensor

def train_contrastive_model(model, tokenizer, df, device='cuda', num_epochs=100, learning_rate=5e-5, margin=1.0, batch_size=32):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    smiles_list = df["SMILES"].tolist()
    bin_labels = df["Bin"].values
    loss_history = []
    for epoch in tqdm(range(num_epochs), desc="Contrastive Training"):
        num_samples = min(1000, len(df))
        indices = torch.randperm(len(df))[:num_samples].numpy()
        batch_indices_list = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        total_loss = 0.0
        num_batches = 0
        for batch_indices in tqdm(batch_indices_list, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_smiles = [smiles_list[i] for i in batch_indices]
            batch_labels = np.array(bin_labels[batch_indices])
            tokens = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(device)
            optimizer.zero_grad()
            _, projected = model(tokens.input_ids, tokens.attention_mask)
            projected_norm = F.normalize(projected, p=1, dim=1)
            dists = torch.cdist(projected_norm, projected_norm, p=1)
            batch_labels_torch = torch.tensor(batch_labels, dtype=torch.long, device=device)
            same_mask = (batch_labels_torch.unsqueeze(0) == batch_labels_torch.unsqueeze(1)).float()
            loss_matrix = same_mask * (dists ** 2) + (1.0 - same_mask) * (torch.clamp(margin - dists, min=0.0) ** 2)
            n = dists.shape[0]
            triu = torch.triu_indices(n, n, offset=1)
            loss_sum = loss_matrix[triu[0], triu[1]].sum()
            num_pairs = triu.shape[1]
            if num_pairs > 0:
                batch_loss_val = loss_sum / num_pairs
                batch_loss_val.backward()
                optimizer.step()
                total_loss += batch_loss_val.item()
                num_batches += 1
        avg = total_loss / num_batches if num_batches > 0 else 0.0
        loss_history.append(avg)
    return model, loss_history

def train_decoder(decoder, contrastive_model, tokenizer, df, char2idx, device='cuda', num_epochs=500, batch_size=32, save_path=None):
    decoder = decoder.to(device)
    decoder.train()
    contrastive_model.eval()
    smiles_list = df["SMILES"].tolist()
    all_embeddings = []
    all_targets = []
    for smi in tqdm(smiles_list, desc="Decoder Embeds"):
        with torch.no_grad():
            emb = contrastive_model.get_embedding(smi, tokenizer, device)
            all_embeddings.append(emb.reshape((768,)))
        t = smiles_to_tensor(smi, char2idx, max_length=150)
        all_targets.append(t)
    import numpy as np
    all_embeddings = torch.tensor(np.array(all_embeddings), dtype=torch.float).to(device)
    all_targets = torch.stack(all_targets).to(device)
    dataset_size = len(all_embeddings)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    all_losses = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Decoder Training"):
        perm = torch.randperm(dataset_size)
        total_loss = 0
        num_batches = 0
        for i in tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            idx = perm[i:i+batch_size]
            batch_emb = all_embeddings[idx]
            batch_tgt = all_targets[idx]
            optimizer.zero_grad()
            loss = decoder(batch_emb, batch_tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches else 0
        all_losses.append(avg_loss)
    if save_path:
        torch.save(decoder.state_dict(), save_path)
    return decoder, all_losses
