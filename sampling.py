import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors
from scipy.optimize import linear_sum_assignment

def fit_gmm(embeddings, n_components=4, random_state=0):
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=random_state)
    gmm.fit(embeddings)
    return gmm

def _assign_components(gmm, embeddings):
    resp = gmm.predict_proba(embeddings)
    comp = resp.argmax(axis=1)
    return comp, resp

def _hungarian_comp_to_bin(comp, bin_labels, n_components):
    y = np.asarray(bin_labels)
    y0 = y - 1 if y.min() == 1 else y
    n_bins = int(y0.max() + 1)
    cm = np.zeros((n_components, n_bins), dtype=np.int64)
    for p, t in zip(comp, y0):
        cm[int(p), int(t)] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {int(r): int(c) + (1 if y.min() == 1 else 0) for r, c in zip(row_ind, col_ind)}
    return mapping

def _lipinski_pass(smiles_arr):
    mols = list(map(Chem.MolFromSmiles, smiles_arr))
    valid = np.array([m is not None for m in mols])
    mols = np.array(mols, dtype=object)
    mw = np.zeros(len(mols))
    hbd = np.zeros(len(mols), dtype=int)
    hba = np.zeros(len(mols), dtype=int)
    logp = np.zeros(len(mols))
    idx = np.where(valid)[0]
    if idx.size > 0:
        sub = mols[idx]
        mw[idx] = np.array([Descriptors.MolWt(m) for m in sub])
        hbd[idx] = np.array([Lipinski.NumHDonors(m) for m in sub])
        hba[idx] = np.array([Lipinski.NumHAcceptors(m) for m in sub])
        logp[idx] = np.array([Descriptors.MolLogP(m) for m in sub])
    mask = valid & (mw < 500.0) & (hbd <= 5) & (hba <= 10) & (logp < 5.0)
    return mask

def _validity_pass(smiles_arr):
    mols = list(map(Chem.MolFromSmiles, smiles_arr))
    valid = np.array([m is not None for m in mols])
    return valid

def _decode_batch(decoder, idx2char, latent_np, temperature=1.0, device='cuda'):
    with torch.no_grad():
        z = torch.tensor(latent_np, dtype=torch.float32, device=device)
        out = decoder.generate_batch(z, idx2char, max_length=150, temperature=temperature, device=device)
    return np.array(out, dtype=object)

def gmm_sampling(gmm, decoder, idx2char, df, embeddings, n_generate=200, max_attempts=20000, temperature=1.0, device='cuda', comp_to_bin=None):
    comp, resp = _assign_components(gmm, embeddings)
    uniq = np.unique(comp)
    out = {}
    dim = embeddings.shape[1]
    for c in uniq:
        mean = gmm.means_[c]
        std = np.sqrt(gmm.covariances_[c])
        need = n_generate
        batch = min(4096, max(512, n_generate))
        smi_list = []
        tried = 0
        seen = set()
        while need > 0 and tried < max_attempts:
            n = min(batch, need)
            noise = np.random.randn(n, dim) * std
            latent = mean + noise
            decoded = _decode_batch(decoder, idx2char, latent, temperature=temperature, device=device)
            valid_mask = _validity_pass(decoded)
            dec = decoded[valid_mask]
            for s in dec:
                if s and s not in seen:
                    smi_list.append(s)
                    seen.add(s)
                    need -= 1
                    if need == 0:
                        break
            tried += n
        key_bin = comp_to_bin.get(int(c), int(c)+1) if comp_to_bin is not None else int(c)+1
        key = f"bin_{int(key_bin)}"
        out[key] = {"gmm_component_number": int(c), "total_smiles_sampled": len(smi_list), "smiles": smi_list}
    return out

def knn_sampling(gmm, decoder, idx2char, df, embeddings, n_generate=200, k=20, top_p=0.2, alpha_noise=0.2, temperature=1.0, device='cuda', comp_to_bin=None):
    comp, resp = _assign_components(gmm, embeddings)
    uniq = np.unique(comp)
    out = {}
    dim = embeddings.shape[1]
    smiles_arr = df["SMILES"].to_numpy()
    for c in uniq:
        mask_c = comp == c
        E = embeddings[mask_c]
        S = smiles_arr[mask_c]
        if E.shape[0] == 0:
            key_bin = comp_to_bin.get(int(c), int(c)+1) if comp_to_bin is not None else int(c)+1
            out[f"bin_{int(key_bin)}"] = {"gmm_component_number": int(c), "total_smiles_sampled": 0, "smiles": []}
            continue
        lip = _lipinski_pass(S)
        score = lip.astype(float)
        if score.sum() == 0:
            score = np.ones_like(score, dtype=float)
        q = max(1, int(np.ceil(top_p * E.shape[0])))
        order = np.argsort(-score)[:q]
        seeds = E[order]
        k_eff = min(k, E.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(E)
        all_dist = nbrs.kneighbors(E, return_distance=True)[0]
        sigma_vec = (np.square(all_dist).mean(axis=1)) ** 0.5
        idx_unique = np.unique(nbrs.kneighbors(seeds, return_distance=False).reshape(-1))
        base = E[idx_unique]
        sigma_base = sigma_vec[idx_unique]
        need = n_generate
        batch = min(4096, max(512, n_generate))
        smi_list = []
        seen = set()
        while need > 0:
            m = min(batch, max(need, batch))
            pick_idx = np.random.randint(0, base.shape[0], size=m)
            pick = base[pick_idx]
            sigma_sel = sigma_base[pick_idx].reshape(-1, 1)
            noise = np.random.randn(m, dim) * (alpha_noise * sigma_sel)
            latent = pick + noise
            decoded = _decode_batch(decoder, idx2char, latent, temperature=temperature, device=device)
            valid_mask = _lipinski_pass(decoded)
            dec = decoded[valid_mask]
            for s in dec:
                if s and s not in seen:
                    smi_list.append(s)
                    seen.add(s)
                    need -= 1
                    if need == 0:
                        break
            if m > n_generate:
                break
        key_bin = comp_to_bin.get(int(c), int(c)+1) if comp_to_bin is not None else int(c)+1
        key = f"bin_{int(key_bin)}"
        out[key] = {"gmm_component_number": int(c), "total_smiles_sampled": len(smi_list), "smiles": smi_list}
    return out

def generate_smiles_with_mapping(gmm, decoder, idx2char, df, embeddings, n_generate=200, max_attempts=20000, temperature=1.0, device='cuda', sampling_mode="unconditional", k=64, top_p=0.2, alpha_noise=0.2, align_bins=True):
    comp_to_bin = None
    if align_bins and ("Bin" in df.columns):
        comp_tmp, _ = _assign_components(gmm, embeddings)
        comp_to_bin = _hungarian_comp_to_bin(comp_tmp, df["Bin"].values, gmm.n_components)
    if sampling_mode == "hill_climbing":
        return knn_sampling(gmm, decoder, idx2char, df, embeddings, n_generate=n_generate, k=k, top_p=top_p, alpha_noise=alpha_noise, temperature=temperature, device=device, comp_to_bin=comp_to_bin)
    return gmm_sampling(gmm, decoder, idx2char, df, embeddings, n_generate=n_generate, max_attempts=max_attempts, temperature=temperature, device=device, comp_to_bin=comp_to_bin)
