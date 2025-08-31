import os
import argparse
import torch
from pathlib import Path
from models import ContrastiveSMILESModel, SMILESDecoder
from utils import (
    set_seed, load_data, load_excel_data, map_kinase_classes, create_bins,
    load_chemberta, extract_embeddings_contrastive, visualize_embeddings,
    build_smiles_vocab, save_contrastive_model, load_contrastive_model,
    save_gmm, load_gmm, save_decoder, load_decoder, save_vocabulary, load_vocabulary,
    save_generated_smiles
)
from training import train_contrastive_model, train_decoder
from sampling import fit_gmm, generate_smiles_with_mapping

def parse_args():
    p = argparse.ArgumentParser(description="Unified IC50/Kinase Pipeline with Specialized Sampling")
    p.add_argument("--dataset", type=str, choices=["ic50", "kinase"], default="ic50")
    p.add_argument("--data_path", type=str, default="IC50_log_transformed_combine.csv")
    p.add_argument("--ic50_column", type=str, default="IC50_log")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--num_bins", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contrastive_epochs", type=int, default=1000)
    p.add_argument("--decoder_epochs", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_generate", type=int, default=2500)
    p.add_argument("--max_attempts", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling_mode", type=str, choices=["unconditional", "hill_climbing"], default="unconditional")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--top_p", type=float, default=0.2)
    p.add_argument("--alpha_noise", type=float, default=0.2)
    p.add_argument("--train_contrastive", action="store_true")
    p.add_argument("--train_decoder", action="store_true")
    p.add_argument("--fit_gmm", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_umap", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    models_root = Path(args.models_dir) / args.dataset
    results_root = Path(args.results_dir) / args.dataset
    models_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    contrastive_model_dir = models_root / "contrastive"
    decoder_path = models_root / "decoder.pt"
    gmm_path = models_root / "gmm.pkl"
    vocab_dir = models_root / "vocab"
    generated_smiles_path = results_root / "generated_smiles.pkl"
    if args.dataset == "ic50":
        df = load_data(args.data_path)
        df = create_bins(df, args.ic50_column, num_bins=args.num_bins)
    else:
        df = load_excel_data(args.data_path)
        df = map_kinase_classes(df, class_column='Class')
        args.num_bins = 4
    all_smiles = df["SMILES"].tolist()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_ready = (vocab_dir / "char2idx.pkl").exists()
    if vocab_ready:
        char2idx, idx2char = load_vocabulary(str(vocab_dir))
    else:
        char2idx, idx2char = build_smiles_vocab(all_smiles)
        save_vocabulary(char2idx, idx2char, str(vocab_dir))
    base_bert_model, base_tokenizer = load_chemberta(device=args.device)
    contrastive_ready = (contrastive_model_dir / "contrastive_model.pt").exists()
    if args.train_contrastive or not contrastive_ready:
        contrastive_model = ContrastiveSMILESModel(base_bert_model)
        contrastive_model, _ = train_contrastive_model(
            contrastive_model, base_tokenizer, df, device=args.device,
            num_epochs=args.contrastive_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size
        )
        contrastive_model_dir.mkdir(parents=True, exist_ok=True)
        save_contrastive_model(contrastive_model, base_tokenizer, str(contrastive_model_dir))
    else:
        contrastive_model, _ = load_contrastive_model(base_bert_model, str(contrastive_model_dir), device=args.device)
    contrastive_embeddings = extract_embeddings_contrastive(df, contrastive_model, base_tokenizer, device=args.device)
    if not args.skip_umap:
        visualize_embeddings(contrastive_embeddings, df["Bin"], "Contrastive Embeddings (UMAP)", save_path=None)
    gmm_ready = gmm_path.exists()
    if args.fit_gmm or not gmm_ready:
        gmm = fit_gmm(contrastive_embeddings, n_components=args.num_bins)
        save_gmm(gmm, str(gmm_path))
    else:
        gmm = load_gmm(str(gmm_path))
    decoder_ready = decoder_path.exists()
    if args.train_decoder or not decoder_ready:
        decoder = SMILESDecoder(
            vocab_size=len(char2idx),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            bert_embed_dim=768,
            max_length=150,
            n_layers=args.n_layers,
            dropout=args.dropout,
            device=args.device
        )
        decoder, _ = train_decoder(
            decoder, contrastive_model, base_tokenizer, df, char2idx,
            device=args.device, num_epochs=args.decoder_epochs, batch_size=args.batch_size, save_path=str(decoder_path)
        )
    else:
        decoder = load_decoder(
            vocab_size=len(char2idx),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            bert_embed_dim=768,
            max_length=150,
            n_layers=args.n_layers,
            dropout=args.dropout,
            save_path=str(decoder_path),
            device=args.device
        )
    if args.generate:
        generated_dict = generate_smiles_with_mapping(
            gmm, decoder, idx2char, df, contrastive_embeddings,
            n_generate=args.n_generate, max_attempts=args.max_attempts, temperature=args.temperature,
            device=args.device, sampling_mode=args.sampling_mode, k=args.k, top_p=args.top_p, alpha_noise=args.alpha_noise
        )
        save_generated_smiles(generated_dict, str(generated_smiles_path))
    print("Done.")

if __name__ == "__main__":
    main()
