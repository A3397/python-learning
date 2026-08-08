"""
================================================================================
  PhD Research Code
  Title  : Cross-Domain Sentiment Analysis Using Bio_ClinicalBERT
  Source : Restaurant Reviews (Yelp)
  Target : Drug Reviews (UCI)
  Method : DAPT  →  (1) With Adversarial Training
                    (2) Without Adversarial Training
  Metrics: Accuracy | Precision | Recall | F1 | Support
  XAI    : SHAP Token-Level Explainability
================================================================================

  KAGGLE SETUP
  ─────────────
  1. Notebook → "+ Add Data" → search "drug review dataset" → Add
  2. Notebook → "+ Add Data" → search "yelp dataset"        → Add
  3. Update the paths in SECTION 1 below
  4. Run All
"""

# ================================================================================
# SECTION 1 — DATASET PATHS  ← SIRF YAHAN CHANGE KARO
# ================================================================================

# ── Restaurant (Source Domain) ──────────────────────────────────────────────────
RESTAURANT_PATH   = "/kaggle/input/yelp-dataset/yelp_review.csv"
RESTAURANT_TEXT   = "text"        # column: review text
RESTAURANT_RATING = "stars"       # column: rating 1–5
RESTAURANT_LABEL  = None          # None → auto-convert from rating

# ── Drug Reviews (Target Domain) ────────────────────────────────────────────────
DRUG_TRAIN_PATH   = "/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.tsv"
DRUG_TEST_PATH    = "/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.tsv"
DRUG_TEXT         = "review"      # column: review text
DRUG_RATING       = "rating"      # column: rating 1–10
DRUG_LABEL        = None          # None → auto-convert from rating

# ── Common Dataset Reference ────────────────────────────────────────────────────
# Dataset Name               Kaggle Folder                   Text Col    Rating Col
# Yelp Reviews               yelp-dataset                    text        stars (1-5)
# Amazon Food Reviews        amazon-fine-food-reviews        Text        Score (1-5)
# UCI Drug Reviews           kuc-hackathon-winter-2018       review      rating (1-10)
# Drug Reviews Extended      drug-review-dataset-uci         reviewText  rating (1-10)


# ================================================================================
# SECTION 2 — HYPERPARAMETERS
# ================================================================================

MODEL_NAME   = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS_DAPT  = 3      # Domain Adaptive Pre-Training — runs ONCE only
EPOCHS_TRAIN = 10     # Main training epochs
LR           = 2e-5
LAMBDA_ADV   = 0.1    # Weight of adversarial domain loss
NUM_LABELS   = 3      # 0 = Positive | 1 = Negative | 2 = Neutral
WARMUP_RATIO = 0.1
MAX_SAMPLES  = 10000  # Set None to use full dataset
SEED         = 42
OUTPUT_DIR   = "/kaggle/working/"
LABEL_NAMES  = ["Positive", "Negative", "Neutral"]
DAPT_PATH    = OUTPUT_DIR + "dapt_bioclinical/"


# ================================================================================
# SECTION 3 — IMPORTS & SETUP
# ================================================================================

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    AdamW, get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("  ENVIRONMENT")
print("=" * 70)
print(f"  Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
print(f"  Model   : {MODEL_NAME}")
print(f"  Epochs  : DAPT={EPOCHS_DAPT}  Train={EPOCHS_TRAIN}")
print(f"  Batch   : {BATCH_SIZE}   LR={LR}   Lambda_adv={LAMBDA_ADV}")


# ================================================================================
# SECTION 4 — METRICS UTILITY
#   Prints per-class Precision | Recall | F1-Score | Support
#   and Macro / Weighted averages — line by line
# ================================================================================

def print_metrics(y_true, y_pred, title=""):
    """
    Detailed metrics table — called after every epoch and after final test.
    Shows: Precision, Recall, F1-Score, Support for each class.
    """
    acc       = accuracy_score(y_true, y_pred)
    prec_cls  = precision_score(y_true, y_pred, average=None,       zero_division=0)
    rec_cls   = recall_score(y_true,   y_pred, average=None,        zero_division=0)
    f1_cls    = f1_score(y_true,       y_pred, average=None,        zero_division=0)
    support   = np.bincount(np.array(y_true), minlength=NUM_LABELS)

    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true,   y_pred, average="macro",     zero_division=0)
    f1_mac    = f1_score(y_true,       y_pred, average="macro",     zero_division=0)

    prec_wt   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_wt    = recall_score(y_true,   y_pred, average="weighted",  zero_division=0)
    f1_wt     = f1_score(y_true,       y_pred, average="weighted",  zero_division=0)

    bar = "─" * 65
    print(f"\n{bar}")
    if title:
        print(f"  {title}  |  Accuracy: {acc*100:.2f}%")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(bar)
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<14} {prec_cls[i]:>10.4f} {rec_cls[i]:>10.4f} {f1_cls[i]:>10.4f} {support[i]:>10d}")
    print(bar)
    print(f"  {'Macro Avg':<14} {prec_mac:>10.4f} {rec_mac:>10.4f} {f1_mac:>10.4f} {sum(support):>10d}")
    print(f"  {'Weighted Avg':<14} {prec_wt:>10.4f} {rec_wt:>10.4f} {f1_wt:>10.4f} {sum(support):>10d}")
    print(bar)

    return dict(acc=acc,
                prec_cls=prec_cls, rec_cls=rec_cls, f1_cls=f1_cls,
                prec_mac=prec_mac, rec_mac=rec_mac, f1_mac=f1_mac)


def print_classification_report(y_true, y_pred, title=""):
    print(f"\n{'=' * 65}")
    print(f"  SKLEARN CLASSIFICATION REPORT — {title}")
    print('=' * 65)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))


# ================================================================================
# SECTION 5 — DATA LOADING
# ================================================================================

def rating_to_label_rest(r):
    """Convert 1–5 star rating → sentiment label."""
    r = float(r)
    if r >= 4:    return 0   # Positive
    elif r <= 2:  return 1   # Negative
    else:         return 2   # Neutral  (3 stars)


def rating_to_label_drug(r):
    """Convert 1–10 drug rating → sentiment label."""
    r = float(r)
    if r >= 7:    return 0   # Positive
    elif r <= 4:  return 1   # Negative
    else:         return 2   # Neutral  (5–6)


def load_domain_data(path, text_col, label_col, rating_col,
                     rating_fn, domain_name, max_samples):
    """
    Universal CSV/TSV loader with automatic label conversion.
    Validates column names and raises informative errors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  [ERROR] File not found: {path}\n"
            f"  → Kaggle → '+ Add Data' → add the dataset\n"
            f"  → Copy the exact path into SECTION 1\n"
        )

    sep = "\t" if path.endswith(".tsv") else ","
    df  = pd.read_csv(path, sep=sep, on_bad_lines="skip", low_memory=False)
    print(f"\n  [{domain_name}] Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    if text_col not in df.columns:
        raise ValueError(
            f"  [ERROR] text_col='{text_col}' not found.\n"
            f"  Available: {list(df.columns)}\n"
            f"  → Update RESTAURANT_TEXT or DRUG_TEXT in SECTION 1\n"
        )

    df = df.rename(columns={text_col: "text"})

    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    elif rating_col and rating_col in df.columns:
        df["label"] = pd.to_numeric(df[rating_col], errors="coerce").apply(
            lambda x: rating_fn(x) if pd.notnull(x) else np.nan
        )
    else:
        raise ValueError(
            f"  [ERROR] Neither label_col='{label_col}' nor "
            f"rating_col='{rating_col}' found.\n"
            f"  Available columns: {list(df.columns)}\n"
        )

    df["domain"] = domain_name
    df = df[["text", "label", "domain"]].dropna()
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)

    dist = df["label"].value_counts().sort_index()
    print(f"         Positive: {dist.get(0,0):>5,}  "
          f"Negative: {dist.get(1,0):>5,}  "
          f"Neutral: {dist.get(2,0):>5,}")
    return df


print("\n" + "=" * 70)
print("  SECTION 5 — LOADING DATASETS")
print("=" * 70)

restaurant_df = load_domain_data(
    RESTAURANT_PATH, RESTAURANT_TEXT, RESTAURANT_LABEL,
    RESTAURANT_RATING, rating_to_label_rest, "Restaurant", MAX_SAMPLES,
)

drugs_df = load_domain_data(
    DRUG_TRAIN_PATH, DRUG_TEXT, DRUG_LABEL,
    DRUG_RATING, rating_to_label_drug, "Drugs", MAX_SAMPLES,
)

# ── Train / Val / Test splits ──────────────────────────────────────────────────
rest_train, r_tmp    = train_test_split(restaurant_df, test_size=0.3,
                                        random_state=SEED, stratify=restaurant_df["label"])
rest_val, rest_test  = train_test_split(r_tmp, test_size=0.5,
                                        random_state=SEED, stratify=r_tmp["label"])

drug_train, d_tmp    = train_test_split(drugs_df, test_size=0.3,
                                        random_state=SEED, stratify=drugs_df["label"])
drug_val, drug_test  = train_test_split(d_tmp, test_size=0.5,
                                        random_state=SEED, stratify=d_tmp["label"])

print(f"\n  Restaurant → Train:{len(rest_train):,}  Val:{len(rest_val):,}  Test:{len(rest_test):,}")
print(f"  Drugs      → Train:{len(drug_train):,}  Val:{len(drug_val):,}  Test:{len(drug_test):,}")


# ================================================================================
# SECTION 6 — TOKENIZER + PYTORCH DATASET
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 6 — TOKENIZER + DATASET CLASS")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Loaded tokenizer: {MODEL_NAME}")


class SentimentDataset(Dataset):
    def __init__(self, df, domain_id=0):
        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.domain_id = domain_id

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            "domain":         torch.tensor(self.domain_id,   dtype=torch.long),
        }


def make_loader(df, domain_id, shuffle=True):
    return DataLoader(
        SentimentDataset(df, domain_id),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=2, pin_memory=True,
    )


rest_train_loader = make_loader(rest_train, 0,        shuffle=True)
rest_val_loader   = make_loader(rest_val,   0,        shuffle=False)
rest_test_loader  = make_loader(rest_test,  0,        shuffle=False)
drug_train_loader = make_loader(drug_train, 1,        shuffle=True)
drug_val_loader   = make_loader(drug_val,   1,        shuffle=False)
drug_test_loader  = make_loader(drug_test,  1,        shuffle=False)


# ================================================================================
# SECTION 7 — DAPT  (Domain Adaptive Pre-Training)
#
#   • Trains Bio_ClinicalBERT on drug-domain text via Masked Language Modeling
#   • Runs ONLY ONCE — saves adapted weights to DAPT_PATH
#   • If already saved, simply loads and skips training
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 7 — DAPT  (Domain Adaptive Pre-Training)")
print("  Runs ONCE on drug-domain text via MLM.  Saved → reused.")
print("=" * 70)


class DAPTDataset(Dataset):
    """MLM Dataset for DAPT — randomly masks 15% of tokens."""
    def __init__(self, texts, mlm_prob=0.15):
        self.texts = texts
        self.prob  = mlm_prob

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0).clone()
        lbl = ids.clone()
        mask_positions = torch.bernoulli(torch.full(ids.shape, self.prob)).bool()
        for sp_id in tokenizer.all_special_ids:
            mask_positions[ids == sp_id] = False
        lbl[~mask_positions] = -100                          # only masked positions contribute to loss
        ids[mask_positions]  = tokenizer.mask_token_id
        return {
            "input_ids":      ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         lbl,
        }


dapt_epoch_losses = []   # tracked for plotting

if os.path.exists(DAPT_PATH + "config.json"):
    # ── Already trained — skip DAPT ───────────────────────────────────────────
    print(f"  ✓  DAPT model already exists at: {DAPT_PATH}")
    print("     Skipping re-training.  Loaded saved weights.")

else:
    # ── Run DAPT ───────────────────────────────────────────────────────────────
    print(f"  Starting DAPT for {EPOCHS_DAPT} epochs on drug-domain text …")
    dapt_model  = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    dapt_loader = DataLoader(
        DAPTDataset(drugs_df["text"].tolist()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    dapt_opt = AdamW(dapt_model.parameters(), lr=3e-5)

    for ep in range(1, EPOCHS_DAPT + 1):
        dapt_model.train()
        ep_loss = 0.0

        for batch in tqdm(dapt_loader, desc=f"  DAPT Epoch {ep}/{EPOCHS_DAPT}"):
            dapt_opt.zero_grad()
            out = dapt_model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            dapt_opt.step()
            ep_loss += out.loss.item()

        avg_loss = ep_loss / len(dapt_loader)
        dapt_epoch_losses.append(avg_loss)
        print(f"  DAPT Epoch {ep}/{EPOCHS_DAPT}  →  MLM Loss: {avg_loss:.4f}")

    os.makedirs(DAPT_PATH, exist_ok=True)
    dapt_model.save_pretrained(DAPT_PATH)
    tokenizer.save_pretrained(DAPT_PATH)
    del dapt_model
    torch.cuda.empty_cache()
    print(f"\n  ✓  DAPT complete.  Adapted model saved to: {DAPT_PATH}")


# ================================================================================
# SECTION 8 — MODEL ARCHITECTURE
#
#   Encoder : Bio_ClinicalBERT (DAPT-adapted)
#   Head 1  : Sentiment Classifier  (3-class softmax)
#   Head 2  : Domain Discriminator  (2-class, via Gradient Reversal Layer)
#             → Used only in adversarial training
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 8 — MODEL ARCHITECTURE")
print("=" * 70)


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class CrossDomainSentimentModel(nn.Module):
    """
    Bio_ClinicalBERT + two task heads:
      1. sentiment_head  → predicts Positive / Negative / Neutral
      2. domain_head     → predicts Restaurant / Drug (via GRL for adversarial)
    """
    def __init__(self, bert_path, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        h = self.bert.config.hidden_size        # 768 for BERT-base

        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_LABELS),
        )

        self.grl = GradientReversalLayer(alpha=LAMBDA_ADV)
        self.domain_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask):
        cls_token   = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0, :]
        sent_logits = self.sentiment_head(cls_token)
        dom_logits  = self.domain_head(self.grl(cls_token))
        return sent_logits, dom_logits, cls_token


criterion_sent = nn.CrossEntropyLoss()
criterion_dom  = nn.CrossEntropyLoss()
print("  Model defined: Bio_ClinicalBERT + Sentiment Head + Domain Head (GRL)")


# ================================================================================
# SECTION 9 — TRAIN & EVAL FUNCTIONS
# ================================================================================

def train_with_adversarial(model, src_loader, tgt_loader, optimizer, scheduler):
    """
    One training epoch WITH adversarial domain adaptation.
      src_loader : restaurant (labeled) — contributes sentiment + domain loss
      tgt_loader : drug (unlabeled)     — contributes domain loss only
    Returns: train_loss, train_sent_loss, train_dom_loss, predictions, labels
    """
    model.train()
    total_loss = sent_loss_sum = dom_loss_sum = 0.0
    all_preds = []; all_labels = []
    tgt_iter  = iter(tgt_loader)

    for batch in tqdm(src_loader, desc="    Train", leave=False):
        optimizer.zero_grad()

        # — Source (restaurant) batch —
        s_sent, s_dom, _ = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        loss_sent = criterion_sent(s_sent, batch["label"].to(DEVICE))
        loss_dom  = criterion_dom(s_dom,   batch["domain"].to(DEVICE))

        # — Target (drug) batch — unlabeled, domain only —
        try:    t_batch = next(tgt_iter)
        except: tgt_iter = iter(tgt_loader); t_batch = next(tgt_iter)

        _, t_dom, _ = model(
            t_batch["input_ids"].to(DEVICE),
            t_batch["attention_mask"].to(DEVICE),
        )
        loss_dom += criterion_dom(t_dom, t_batch["domain"].to(DEVICE))

        loss = loss_sent + LAMBDA_ADV * loss_dom
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        total_loss    += loss.item()
        sent_loss_sum += loss_sent.item()
        dom_loss_sum  += loss_dom.item()
        all_preds.extend(torch.argmax(s_sent, 1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    n = len(src_loader)
    return total_loss/n, sent_loss_sum/n, dom_loss_sum/n, all_preds, all_labels


def train_without_adversarial(model, loader, optimizer, scheduler):
    """
    One training epoch WITHOUT adversarial — plain sentiment fine-tuning.
    Returns: train_loss, predictions, labels
    """
    model.train()
    total_loss = 0.0
    all_preds = []; all_labels = []

    for batch in tqdm(loader, desc="    Train", leave=False):
        optimizer.zero_grad()
        logits, _, _ = model(batch["input_ids"].to(DEVICE),
                             batch["attention_mask"].to(DEVICE))
        loss = criterion_sent(logits, batch["label"].to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    return total_loss / len(loader), all_preds, all_labels


def evaluate(model, loader):
    """
    Evaluate model on any loader.
    Returns: loss, predictions, true_labels
    """
    model.eval()
    total_loss = 0.0
    all_preds = []; all_labels = []

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = model(batch["input_ids"].to(DEVICE),
                                 batch["attention_mask"].to(DEVICE))
            total_loss += criterion_sent(logits, batch["label"].to(DEVICE)).item()
            all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    return total_loss / len(loader), all_preds, all_labels


# ================================================================================
# SECTION 10 — MODEL 1: WITH ADVERSARIAL TRAINING
#
#   • Starts from DAPT-adapted Bio_ClinicalBERT
#   • Source domain (restaurant) → sentiment + domain loss
#   • Target domain (drug)       → domain adversarial loss only
#   • Evaluated on drug val set every epoch
#   • Tracks: train_loss, test_loss, train_acc, test_acc per epoch
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 10 — MODEL 1: WITH ADVERSARIAL TRAINING")
print("  Base: DAPT-adapted Bio_ClinicalBERT")
print("  Train on: Restaurant (source)  |  Eval on: Drugs (target)")
print("=" * 70)

model_adv    = CrossDomainSentimentModel(DAPT_PATH).to(DEVICE)
opt_adv      = AdamW(model_adv.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(rest_train_loader) * EPOCHS_TRAIN
warmup_steps = int(total_steps * WARMUP_RATIO)
sched_adv    = get_linear_schedule_with_warmup(opt_adv, warmup_steps, total_steps)

# ── History arrays (one value per epoch) ─────────────────────────────────────
adv_train_loss = []    # training loss     per epoch
adv_test_loss  = []    # validation loss   per epoch  (on drug domain)
adv_train_acc  = []    # training accuracy per epoch
adv_test_acc   = []    # validation accuracy per epoch (on drug domain)
adv_train_f1   = []
adv_test_f1    = []
adv_test_prec  = []
adv_test_rec   = []

best_adv_acc = 0.0

for epoch in range(1, EPOCHS_TRAIN + 1):

    # ── TRAINING ──────────────────────────────────────────────────────────────
    tr_loss, tr_s_loss, tr_d_loss, tr_preds, tr_labels = train_with_adversarial(
        model_adv, rest_train_loader, drug_train_loader, opt_adv, sched_adv,
    )
    tr_metrics = print_metrics(
        tr_labels, tr_preds,
        title=f"MODEL 1 (Adversarial)  TRAINING  Epoch {epoch}/{EPOCHS_TRAIN}",
    )

    # ── TESTING on drug domain ─────────────────────────────────────────────────
    te_loss, te_preds, te_labels = evaluate(model_adv, drug_val_loader)
    te_metrics = print_metrics(
        te_labels, te_preds,
        title=f"MODEL 1 (Adversarial)  TESTING   Epoch {epoch}/{EPOCHS_TRAIN}",
    )

    # ── Store per-epoch values ─────────────────────────────────────────────────
    adv_train_loss.append(tr_loss)
    adv_test_loss.append(te_loss)
    adv_train_acc.append(tr_metrics["acc"])
    adv_test_acc.append(te_metrics["acc"])
    adv_train_f1.append(tr_metrics["f1_mac"])
    adv_test_f1.append(te_metrics["f1_mac"])
    adv_test_prec.append(te_metrics["prec_mac"])
    adv_test_rec.append(te_metrics["rec_mac"])

    print(f"\n  ── Epoch {epoch:02d} Summary ──  "
          f"Train Loss: {tr_loss:.4f}  Test Loss: {te_loss:.4f}  "
          f"Train Acc: {tr_metrics['acc']*100:.2f}%  "
          f"Test Acc: {te_metrics['acc']*100:.2f}%")

    # ── Save best model ────────────────────────────────────────────────────────
    if te_metrics["acc"] > best_adv_acc:
        best_adv_acc = te_metrics["acc"]
        torch.save(model_adv.state_dict(), OUTPUT_DIR + "best_adv.pt")
        print(f"  ★ Best adversarial model saved  (test_acc = {best_adv_acc*100:.2f}%)")


# ================================================================================
# SECTION 11 — MODEL 2: WITHOUT ADVERSARIAL TRAINING (Baseline)
#
#   • Same DAPT-adapted base as Model 1
#   • Trains only on restaurant sentiment — no domain adversarial loss
#   • Evaluated on drug val set every epoch
#   • Tracks: train_loss, test_loss, train_acc, test_acc per epoch
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 11 — MODEL 2: WITHOUT ADVERSARIAL TRAINING  (Baseline)")
print("  Base: DAPT-adapted Bio_ClinicalBERT (same starting weights)")
print("  Train on: Restaurant (source)  |  Eval on: Drugs (target)")
print("=" * 70)

model_plain  = CrossDomainSentimentModel(DAPT_PATH).to(DEVICE)
opt_plain    = AdamW(model_plain.parameters(), lr=LR, weight_decay=0.01)
sched_plain  = get_linear_schedule_with_warmup(opt_plain, warmup_steps, total_steps)

# ── History arrays ────────────────────────────────────────────────────────────
plain_train_loss = []
plain_test_loss  = []
plain_train_acc  = []
plain_test_acc   = []
plain_train_f1   = []
plain_test_f1    = []
plain_test_prec  = []
plain_test_rec   = []

best_plain_acc = 0.0

for epoch in range(1, EPOCHS_TRAIN + 1):

    # ── TRAINING ──────────────────────────────────────────────────────────────
    tr_loss, tr_preds, tr_labels = train_without_adversarial(
        model_plain, rest_train_loader, opt_plain, sched_plain,
    )
    tr_metrics = print_metrics(
        tr_labels, tr_preds,
        title=f"MODEL 2 (Baseline)     TRAINING  Epoch {epoch}/{EPOCHS_TRAIN}",
    )

    # ── TESTING on drug domain ─────────────────────────────────────────────────
    te_loss, te_preds, te_labels = evaluate(model_plain, drug_val_loader)
    te_metrics = print_metrics(
        te_labels, te_preds,
        title=f"MODEL 2 (Baseline)     TESTING   Epoch {epoch}/{EPOCHS_TRAIN}",
    )

    # ── Store per-epoch values ─────────────────────────────────────────────────
    plain_train_loss.append(tr_loss)
    plain_test_loss.append(te_loss)
    plain_train_acc.append(tr_metrics["acc"])
    plain_test_acc.append(te_metrics["acc"])
    plain_train_f1.append(tr_metrics["f1_mac"])
    plain_test_f1.append(te_metrics["f1_mac"])
    plain_test_prec.append(te_metrics["prec_mac"])
    plain_test_rec.append(te_metrics["rec_mac"])

    print(f"\n  ── Epoch {epoch:02d} Summary ──  "
          f"Train Loss: {tr_loss:.4f}  Test Loss: {te_loss:.4f}  "
          f"Train Acc: {tr_metrics['acc']*100:.2f}%  "
          f"Test Acc: {te_metrics['acc']*100:.2f}%")

    if te_metrics["acc"] > best_plain_acc:
        best_plain_acc = te_metrics["acc"]
        torch.save(model_plain.state_dict(), OUTPUT_DIR + "best_plain.pt")
        print(f"  ★ Best baseline model saved  (test_acc = {best_plain_acc*100:.2f}%)")


# ================================================================================
# SECTION 12 — FINAL TEST EVALUATION
#   Load best weights, evaluate on held-out drug test set
#   Print full Precision | Recall | F1 | Support + sklearn classification_report
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 12 — FINAL TEST EVALUATION  (held-out drug test set)")
print("=" * 70)

model_adv.load_state_dict(torch.load(OUTPUT_DIR + "best_adv.pt",   map_location=DEVICE))
model_plain.load_state_dict(torch.load(OUTPUT_DIR + "best_plain.pt", map_location=DEVICE))

_, final_preds_adv,   final_labels = evaluate(model_adv,   drug_test_loader)
_, final_preds_plain, _            = evaluate(model_plain, drug_test_loader)

print("\n  ─────────────────────────────────────")
print("  Model 1 — WITH Adversarial Training")
print("  ─────────────────────────────────────")
adv_final   = print_metrics(final_labels, final_preds_adv,
                             title="MODEL 1 FINAL TEST — WITH ADVERSARIAL")
print_classification_report(final_labels, final_preds_adv,
                             title="MODEL 1 — WITH ADVERSARIAL TRAINING")

print("\n  ─────────────────────────────────────")
print("  Model 2 — WITHOUT Adversarial Training")
print("  ─────────────────────────────────────")
plain_final = print_metrics(final_labels, final_preds_plain,
                             title="MODEL 2 FINAL TEST — WITHOUT ADVERSARIAL")
print_classification_report(final_labels, final_preds_plain,
                             title="MODEL 2 — WITHOUT ADVERSARIAL TRAINING")


# ================================================================================
# SECTION 13 — SHAP EXPLAINABILITY
#   Token-level feature importance using SHAP Explainer
#   Shows which token positions drive each sentiment class
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 13 — SHAP EXPLAINABILITY")
print("  Token-level importance via SHAP on Model 1 (adversarial)")
print("=" * 70)


def model_predict_proba(texts):
    """Wrapper: text list → probability array  (used by SHAP)."""
    model_adv.eval()
    probs_all = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = list(texts[i: i + BATCH_SIZE])
        enc   = tokenizer(chunk, max_length=MAX_LEN, padding="max_length",
                          truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits, _, _ = model_adv(enc["input_ids"].to(DEVICE),
                                     enc["attention_mask"].to(DEVICE))
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs_all)


shap_bg      = drug_test["text"].tolist()[:20]   # background reference set
shap_samples = drug_test["text"].tolist()[:40]   # samples to explain

print(f"  Background samples : {len(shap_bg)}")
print(f"  Evaluation samples : {len(shap_samples)}")

explainer    = shap.Explainer(model_predict_proba, shap_bg)
shap_values  = explainer(shap_samples)            # shape: (40, MAX_LEN, 3)

# Per-class mean |SHAP| over evaluation samples  → shape: (MAX_LEN, 3)
shap_per_class = np.abs(shap_values.values).mean(axis=0)

# Overall importance across all classes  → shape: (MAX_LEN,)
shap_overall   = shap_per_class.mean(axis=1)

# Print per-class top-10 token positions
print("\n  Top-10 important token positions per class:")
for ci, cname in enumerate(LABEL_NAMES):
    top10 = np.argsort(shap_per_class[:, ci])[-10:][::-1]
    print(f"\n  [{cname}]")
    print(f"  {'Position':<14} {'Mean |SHAP|':>14}")
    print(f"  {'─' * 30}")
    for pos in top10:
        print(f"  tok_pos_{pos:<6}  {shap_per_class[pos, ci]:>14.6f}")


# ================================================================================
# SECTION 14 — PLOTS
#
#   Figure 1 : Accuracy vs Epoch
#              — Training Accuracy (both models)
#              — Testing  Accuracy (both models)
#              → 2 subplots side by side (Model1 | Model2)
#
#   Figure 2 : Loss vs Epoch
#              — Training Loss
#              — Testing Loss
#              → 2 subplots side by side (Model1 | Model2)
#
#   Figure 3 : Precision | Recall | F1 vs Epoch  (val / test)
#   Figure 4 : Confusion Matrices
#   Figure 5 : Per-class F1 | Precision | Recall bar charts
#   Figure 6 : SHAP — overall + per-class token importance
#   Figure 7 : DAPT MLM loss  (if DAPT ran this session)
# ================================================================================

print("\n" + "=" * 70)
print("  SECTION 14 — GENERATING PLOTS")
print("=" * 70)

ep = list(range(1, EPOCHS_TRAIN + 1))
plt.style.use("seaborn-v0_8-whitegrid")

# ── Colors ────────────────────────────────────────────────────────────────────
TRAIN_COLOR = "#2E86AB"   # blue  — training lines
TEST_COLOR  = "#E84855"   # red   — testing lines
ADV_COLOR   = "#2E86AB"   # blue  — adversarial model
PLAIN_COLOR = "#E84855"   # red   — baseline model
TARGET_LINE = "#2DC653"   # green — 90% target line
DAPT_COLOR  = "#F4A261"   # orange


# ── FIGURE 1 — Accuracy vs Epoch ──────────────────────────────────────────────
#   Left plot  : Model 1 (Adversarial)  — Train Acc + Test Acc
#   Right plot : Model 2 (Baseline)     — Train Acc + Test Acc

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Accuracy vs. Epoch", fontsize=15, fontweight="bold")

# Model 1 — Adversarial
axes1[0].plot(ep, [v*100 for v in adv_train_acc],
              "o-", color=TRAIN_COLOR, lw=2.2, ms=6, label="Training Accuracy")
axes1[0].plot(ep, [v*100 for v in adv_test_acc],
              "s--", color=TEST_COLOR,  lw=2.2, ms=6, label="Testing Accuracy")
axes1[0].axhline(90, color=TARGET_LINE, ls=":", lw=1.8, alpha=0.85, label="90% Target")
axes1[0].set_title("Model 1 — With Adversarial Training", fontweight="bold")
axes1[0].set_xlabel("Epoch"); axes1[0].set_ylabel("Accuracy (%)")
axes1[0].legend(); axes1[0].set_ylim([0, 105])
axes1[0].set_xticks(ep)

# Model 2 — Baseline (no adversarial)
axes1[1].plot(ep, [v*100 for v in plain_train_acc],
              "o-", color=TRAIN_COLOR, lw=2.2, ms=6, label="Training Accuracy")
axes1[1].plot(ep, [v*100 for v in plain_test_acc],
              "s--", color=TEST_COLOR,  lw=2.2, ms=6, label="Testing Accuracy")
axes1[1].axhline(90, color=TARGET_LINE, ls=":", lw=1.8, alpha=0.85, label="90% Target")
axes1[1].set_title("Model 2 — Without Adversarial Training", fontweight="bold")
axes1[1].set_xlabel("Epoch"); axes1[1].set_ylabel("Accuracy (%)")
axes1[1].legend(); axes1[1].set_ylim([0, 105])
axes1[1].set_xticks(ep)

fig1.tight_layout()
fig1.savefig(OUTPUT_DIR + "fig1_accuracy_vs_epoch.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig1_accuracy_vs_epoch.png  saved")


# ── FIGURE 2 — Loss vs Epoch ───────────────────────────────────────────────────
#   Left plot  : Model 1 — Training Loss + Testing Loss
#   Right plot : Model 2 — Training Loss + Testing Loss

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Loss vs. Epoch", fontsize=15, fontweight="bold")

# Model 1 — Adversarial
axes2[0].plot(ep, adv_train_loss,
              "o-", color=TRAIN_COLOR, lw=2.2, ms=6, label="Training Loss")
axes2[0].plot(ep, adv_test_loss,
              "s--", color=TEST_COLOR,  lw=2.2, ms=6, label="Testing Loss")
axes2[0].set_title("Model 1 — With Adversarial Training", fontweight="bold")
axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Loss")
axes2[0].legend(); axes2[0].set_xticks(ep)

# Model 2 — Baseline
axes2[1].plot(ep, plain_train_loss,
              "o-", color=TRAIN_COLOR, lw=2.2, ms=6, label="Training Loss")
axes2[1].plot(ep, plain_test_loss,
              "s--", color=TEST_COLOR,  lw=2.2, ms=6, label="Testing Loss")
axes2[1].set_title("Model 2 — Without Adversarial Training", fontweight="bold")
axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("Loss")
axes2[1].legend(); axes2[1].set_xticks(ep)

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR + "fig2_loss_vs_epoch.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig2_loss_vs_epoch.png  saved")


# ── FIGURE 3 — Precision | Recall | F1 vs Epoch (test set) ───────────────────

fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6))
fig3.suptitle("Precision | Recall | F1-Score vs. Epoch  (Drug Test Domain)",
              fontsize=14, fontweight="bold")

metrics_pairs = [
    (adv_test_prec,  plain_test_prec,  "Macro Precision"),
    (adv_test_rec,   plain_test_rec,   "Macro Recall"),
    (adv_test_f1,    plain_test_f1,    "Macro F1-Score"),
]

for ax, (adv_vals, plain_vals, ylabel) in zip(axes3, metrics_pairs):
    ax.plot(ep, [v*100 for v in adv_vals],
            "o-", color=ADV_COLOR,   lw=2, ms=6, label="With Adversarial")
    ax.plot(ep, [v*100 for v in plain_vals],
            "s--", color=PLAIN_COLOR, lw=2, ms=6, label="Without Adversarial")
    ax.axhline(90, color=TARGET_LINE, ls=":", lw=1.8, alpha=0.85, label="90% Target")
    ax.set_title(ylabel, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(f"{ylabel} (%)")
    ax.legend(); ax.set_ylim([0, 105]); ax.set_xticks(ep)

fig3.tight_layout()
fig3.savefig(OUTPUT_DIR + "fig3_prec_rec_f1_vs_epoch.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig3_prec_rec_f1_vs_epoch.png  saved")


# ── FIGURE 4 — Confusion Matrices ─────────────────────────────────────────────

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))
fig4.suptitle("Confusion Matrices — Final Test Set (Drug Domain)",
              fontsize=14, fontweight="bold")

sns.heatmap(confusion_matrix(final_labels, final_preds_adv),
            annot=True, fmt="d", cmap="Blues", ax=axes4[0],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5)
axes4[0].set_title("Model 1 — With Adversarial Training", fontweight="bold")
axes4[0].set_xlabel("Predicted Label"); axes4[0].set_ylabel("True Label")

sns.heatmap(confusion_matrix(final_labels, final_preds_plain),
            annot=True, fmt="d", cmap="Reds", ax=axes4[1],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5)
axes4[1].set_title("Model 2 — Without Adversarial Training", fontweight="bold")
axes4[1].set_xlabel("Predicted Label"); axes4[1].set_ylabel("True Label")

fig4.tight_layout()
fig4.savefig(OUTPUT_DIR + "fig4_confusion_matrices.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig4_confusion_matrices.png  saved")


# ── FIGURE 5 — Per-class Precision | Recall | F1 bar charts ───────────────────

fig5, axes5 = plt.subplots(1, 3, figsize=(22, 7))
fig5.suptitle("Per-Class Metrics — Final Test Set (Drug Domain)",
              fontsize=14, fontweight="bold")
x = np.arange(len(LABEL_NAMES))

bar_triples = [
    (adv_final["prec_cls"], plain_final["prec_cls"], "Precision"),
    (adv_final["rec_cls"],  plain_final["rec_cls"],  "Recall"),
    (adv_final["f1_cls"],   plain_final["f1_cls"],   "F1-Score"),
]

for ax, (adv_v, plain_v, metric) in zip(axes5, bar_triples):
    ax.bar(x - 0.2, adv_v*100,   0.38, label="With Adv",    color=ADV_COLOR,   alpha=0.88)
    ax.bar(x + 0.2, plain_v*100, 0.38, label="Without Adv", color=PLAIN_COLOR, alpha=0.88)
    ax.axhline(90, color=TARGET_LINE, ls="--", lw=1.5, alpha=0.8, label="90% Target")
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_title(f"Per-Class {metric}", fontweight="bold")
    ax.set_ylabel(f"{metric} (%)"); ax.legend(); ax.set_ylim([0, 110])

fig5.tight_layout()
fig5.savefig(OUTPUT_DIR + "fig5_perclass_metrics.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig5_perclass_metrics.png  saved")


# ── FIGURE 6 — SHAP Token Importance ──────────────────────────────────────────

fig6, axes6 = plt.subplots(1, 2, figsize=(20, 9))
fig6.suptitle("SHAP Explainability — Bio_ClinicalBERT + Adversarial Model\n"
              "Token Position Importance on Drug Domain Samples",
              fontsize=13, fontweight="bold")

# Overall importance — top 25 positions
top25 = np.argsort(shap_overall)[-25:][::-1]
axes6[0].barh([f"pos_{i}" for i in top25],
              shap_overall[top25], color=ADV_COLOR, alpha=0.82)
axes6[0].axvline(shap_overall[top25].mean(),
                 color="red", ls="--", lw=1.5, alpha=0.7, label="Mean")
axes6[0].set_title("Overall Token Importance\n(Mean |SHAP| across Pos + Neg + Neutral)",
                   fontweight="bold")
axes6[0].set_xlabel("Mean |SHAP Value|"); axes6[0].legend()

# Per-class importance — top 15 positions
top15      = np.argsort(shap_overall)[-15:][::-1]
bar_cols   = [ADV_COLOR, PLAIN_COLOR, DAPT_COLOR]
bar_w      = 0.22
y_base     = np.arange(len(top15))

for ci, (cname, col) in enumerate(zip(LABEL_NAMES, bar_cols)):
    axes6[1].barh(y_base + ci * bar_w,
                  shap_per_class[top15, ci],
                  bar_w, label=cname, color=col, alpha=0.85)

axes6[1].set_yticks(y_base + bar_w)
axes6[1].set_yticklabels([f"pos_{i}" for i in top15])
axes6[1].set_title("Per-Class Token Importance\n(Top 15 Positions)",
                   fontweight="bold")
axes6[1].set_xlabel("Mean |SHAP Value|"); axes6[1].legend()

fig6.tight_layout()
fig6.savefig(OUTPUT_DIR + "fig6_shap.png",
             dpi=180, bbox_inches="tight", facecolor="white")
print("  ✓  fig6_shap.png  saved")


# ── FIGURE 7 — DAPT MLM Loss  (only if DAPT ran this session) ─────────────────

if dapt_epoch_losses:
    fig7, ax7 = plt.subplots(figsize=(9, 5))
    dapt_ep = list(range(1, EPOCHS_DAPT + 1))
    ax7.plot(dapt_ep, dapt_epoch_losses,
             "o-", color=DAPT_COLOR, lw=2.5, ms=9,
             markeredgecolor="white", markeredgewidth=1.5)
    ax7.fill_between(dapt_ep, dapt_epoch_losses, alpha=0.15, color=DAPT_COLOR)
    for i, v in enumerate(dapt_epoch_losses):
        ax7.annotate(f"{v:.4f}", (i+1, v),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10, fontweight="bold")
    ax7.set_title("DAPT — MLM Pre-Training Loss\n(Bio_ClinicalBERT adapted on Drug Domain)",
                  fontsize=12, fontweight="bold")
    ax7.set_xlabel("Epoch"); ax7.set_ylabel("MLM Loss"); ax7.set_xticks(dapt_ep)
    fig7.tight_layout()
    fig7.savefig(OUTPUT_DIR + "fig7_dapt_loss.png",
                 dpi=180, bbox_inches="tight", facecolor="white")
    print("  ✓  fig7_dapt_loss.png  saved")


# ================================================================================
# SECTION 15 — FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n  {'Model':<42} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  90%?")
print(f"  {'─'*65}")
for name, m in [("Bio_ClinicalBERT + DAPT + Adversarial", adv_final),
                ("Bio_ClinicalBERT + DAPT  (no Adversarial)", plain_final)]:
    target = "✓" if m["acc"] >= 0.90 else "✗"
    print(f"  {name:<42} "
          f"{m['acc']*100:>6.2f}% "
          f"{m['prec_mac']*100:>6.2f}% "
          f"{m['rec_mac']*100:>6.2f}% "
          f"{m['f1_mac']*100:>6.2f}%  {target}")

print(f"""
  Saved files → {OUTPUT_DIR}
  ──────────────────────────────────────────────
  fig1_accuracy_vs_epoch.png   Training Acc + Testing Acc per epoch
  fig2_loss_vs_epoch.png       Training Loss + Testing Loss per epoch
  fig3_prec_rec_f1_vs_epoch.png  Precision | Recall | F1 per epoch
  fig4_confusion_matrices.png  Confusion matrices (both models)
  fig5_perclass_metrics.png    Per-class Precision | Recall | F1 bars
  fig6_shap.png                SHAP token importance
  fig7_dapt_loss.png           DAPT MLM pre-training loss
  best_adv.pt                  Best adversarial model weights
  best_plain.pt                Best baseline model weights
  dapt_bioclinical/            DAPT-adapted Bio_ClinicalBERT
""")
print("  PhD-level analysis complete.")
