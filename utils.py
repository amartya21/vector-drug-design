import os
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_excel_data(path):
    xls = pd.ExcelFile(path)
    for name in xls.sheet_names:
        if "Sheet" in name or "data" in name.lower():
            df = xls.parse(name)
            break
    else:
        df = xls.parse(xls.sheet_names[0])
    if "SMILES" not in df.columns:
        raise ValueError("SMILES column missing in Excel.")
    return df

def map_kinase_classes(df, class_column='Class'):
    mapping = {'A': 1, 'I': 2, 'I1/2': 3, 'II': 4}
    df = df.copy()
    df["Bin"] = df[class_column].map(mapping).astype(int)
    return df

def create_bins(df, column_name, num_bins=4):
    df['Bin'] = pd.qcut(df[column_name], q=num_bins, labels=False) + 1
    return df

def load_chemberta(model_name="seyonec/ChemBERTa-zinc-base-v1", device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer

def extract_embeddings(df, model, tokenizer, device='cuda'):
    model.eval()
    embeddings = []
    for smiles in tqdm(df["SMILES"], desc="Extracting embeddings"):
        with torch.no_grad():
            tokens = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            embeddings.append(emb)
    embeddings = np.array(embeddings)
    return embeddings

def extract_embeddings_contrastive(df, contrastive_model, tokenizer, device='cuda'):
    contrastive_model.eval()
    embeddings = []
    for smiles in tqdm(df["SMILES"], desc="Extracting contrastive embeddings"):
        with torch.no_grad():
            emb = contrastive_model.get_embedding(smiles, tokenizer, device)
            embeddings.append(emb)
    embeddings = np.array(embeddings)
    return embeddings

def visualize_embeddings(embeddings, labels, title, save_path=None, dim_reduction="UMAP"):
    if dim_reduction == "UMAP":
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
        reduced = reducer.fit_transform(embeddings)
    df_viz = pd.DataFrame(reduced, columns=["Dim1", "Dim2"])
    df_viz["Bin"] = labels
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_viz, x="Dim1", y="Dim2", hue="Bin", edgecolor="k", alpha=0.7)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="IC50 Bins")
    if save_path:
        pass
    return plt.gcf()

def build_smiles_vocab(smiles_list):
    special_tokens = ["<PAD>", "< SOS >", "<EOS>"]
    unique_chars = set()
    for smi in smiles_list:
        for ch in smi:
            unique_chars.add(ch)
    unique_chars = sorted(list(unique_chars))
    all_tokens = special_tokens + unique_chars
    char2idx = {ch: i for i, ch in enumerate(all_tokens)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char

def smiles_to_tensor(smiles, char2idx, max_length=150):
    tokens = ["< SOS >"] + list(smiles) + ["<EOS>"]
    tokens = tokens[:max_length]
    ids = [char2idx[ch] for ch in tokens if ch in char2idx]
    if len(ids) < max_length:
        ids += [char2idx["<PAD>"]] * (max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def save_contrastive_model(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "contrastive_model.pt"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

def load_contrastive_model(bert_model, save_dir, device='cuda'):
    from models import ContrastiveSMILESModel
    model = ContrastiveSMILESModel(bert_model)
    model.load_state_dict(torch.load(os.path.join(save_dir, "contrastive_model.pt"), map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_dir, "tokenizer"))
    return model, tokenizer

def save_decoder(decoder, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(decoder.state_dict(), save_path)

def load_decoder(vocab_size, embed_dim, hidden_dim, bert_embed_dim=768, max_length=150, n_layers=3, dropout=0.2, save_path=None, device='cuda'):
    from models import SMILESDecoder
    decoder = SMILESDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        bert_embed_dim=bert_embed_dim,
        max_length=max_length,
        n_layers=n_layers,
        dropout=dropout,
        device=device
    )
    if save_path:
        decoder.load_state_dict(torch.load(save_path, map_location=device))
    return decoder

def save_gmm(gmm, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(gmm, f)

def load_gmm(file_path):
    with open(file_path, 'rb') as f:
        gmm = pickle.load(f)
    return gmm

def save_vocabulary(char2idx, idx2char, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "char2idx.pkl"), 'wb') as f:
        pickle.dump(char2idx, f)
    with open(os.path.join(save_dir, "idx2char.pkl"), 'wb') as f:
        pickle.dump(idx2char, f)

def load_vocabulary(save_dir):
    with open(os.path.join(save_dir, "char2idx.pkl"), 'rb') as f:
        char2idx = pickle.load(f)
    with open(os.path.join(save_dir, "idx2char.pkl"), 'rb') as f:
        idx2char = pickle.load(f)
    return char2idx, idx2char

def save_generated_smiles(generated_smiles, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True
                )
    with open(file_path, 'wb') as f:
        pickle.dump(generated_smiles, f)

def load_generated_smiles(file_path):
    with open(file_path, 'rb') as f:
        generated_smiles = pickle.load(f)
    return generated_smiles

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception:
        return None
