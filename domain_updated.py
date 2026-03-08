"""
Multi-Domain Sentiment Analyzer
=================================
Architecture
------------
  Shared RoBERTa encoder  (fine-tuned for all domains jointly)
      │
      ├── DomainAdapter["finance"]  ──►  SentimentHead  →  score ∈ [0,1]
      ├── DomainAdapter["tech"]     ──►  SentimentHead  →  score ∈ [0,1]
      ├── DomainAdapter["business"] ──►  SentimentHead  →  score ∈ [0,1]
      ├── DomainAdapter["politics"] ──►  SentimentHead  →  score ∈ [0,1]
      └── DomainAdapter["law"]      ──►  SentimentHead  →  score ∈ [0,1]

Usage
-----
  # Training
  pipeline = DomainSentimentPipeline(cfg)
  pipeline.train()

  # Inference — you supply the domain, model uses that adapter + head
  pipeline.predict("Fed raises interest rates.", domain="finance")
  pipeline.predict("Apple antitrust case dismissed.", domain="law")
  pipeline.predict("Senate passes AI regulation bill.", domain="politics")

Each domain is trained only on its own data, so the score reflects
sentiment *within that domain's context*, not generic financial sentiment.

Score: 0.0 = most negative  │  0.5 = neutral  │  1.0 = most positive

Install
-------
  pip install torch transformers tqdm pandas

Data format (per domain)
------------------------
  Any .csv / .tsv / .jsonl / .json / .parquet / .txt file.
  The loader auto-detects the text column.
  Files are tagged with a domain when you add them to cfg.data_files:

    cfg.data_files = [
        ("data/finance_phrasebank.jsonl", "finance"),
        ("data/twitter_finance.jsonl",    "finance"),
        ("data/tech_news.jsonl",           "tech"),
        ("data/politics_headlines.jsonl",  "politics"),
        ("data/law_cases.jsonl",           "law"),
        ("data/business_news.jsonl",       "business"),
    ]
"""

from __future__ import annotations

import os
import json
import csv
import math
import random
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ============================================================
# 0.  Configuration
# ============================================================

@dataclass
class Config:
    # --- Models ---
    student_model : str = "roberta-base"

    # --- Domains ---
    # Add or remove domains freely — each gets its own adapter + head
    domains : list[str] = field(default_factory=lambda: [
        "finance", "tech", "business", "politics", "law"
    ])

    # --- Data ---
    # List of (file_path, domain_name) tuples.
    # domain_name must be one of the strings in cfg.domains above.
    # Example:
    #   data_files = [
    #       ("data/finance_phrasebank.jsonl", "finance"),
    #       ("data/tech_news.jsonl",           "tech"),
    #   ]
    data_files : list[tuple[str, str]] = field(default_factory=list)

    max_samples_per_domain : int = 5_000   # cap per domain, not total

    # --- Adapter ---
    bottleneck_size : int = 64

    # --- Training ---
    epochs_frozen      : int   = 5      # train adapters + heads only
    epochs_unfrozen    : int   = 3      # full fine-tune
    batch_size         : int   = 16
    lr_frozen          : float = 3e-4
    lr_unfrozen        : float = 2e-5
    weight_decay       : float = 1e-2
    warmup_ratio       : float = 0.1
    max_length         : int   = 128
    val_split          : float = 0.1
    dropout            : float = 0.15
    grad_clip          : float = 1.0
    early_stop_patience: int   = 3

    # --- Teacher / labelling ---
    temperature           : float = 3.0
    min_teacher_confidence: float = 0.60  # drop sample if teacher is below this
    soft_label_weight     : float = 0.5

    # --- Misc ---
    seed      : int = 42
    device    : str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path : str = "domain_sentiment_model.pt"


# ============================================================
# 1.  Architecture
# ============================================================

class DomainAdapter(nn.Module):
    """Pfeiffer bottleneck adapter — one per domain."""
    def __init__(self, hidden: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(x)))) + x   # residual


class SentimentHead(nn.Module):
    """Regression + classification head — one per domain."""
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),           # neg / neu / pos logits
        )

    def forward(self, x: torch.Tensor):
        return self.reg(x).squeeze(-1), self.cls(x)   # score (B,), logits (B,3)


class MultiDomainModel(nn.Module):
    """
    Shared encoder with N independent domain adapters and sentiment heads.
    At inference time, only the adapter + head for the requested domain runs.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.domains = cfg.domains
        self.encoder = AutoModel.from_pretrained(cfg.student_model)
        H            = self.encoder.config.hidden_size

        self.adapters = nn.ModuleDict({
            d: DomainAdapter(H, cfg.bottleneck_size) for d in cfg.domains
        })
        self.heads = nn.ModuleDict({
            d: SentimentHead(H, cfg.dropout) for d in cfg.domains
        })

    def _encode(self, input_ids, attention_mask) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]          # CLS token  (B, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain: str,
    ):
        """
        Returns (score, logits) using only the adapter + head for `domain`.
        This is the key design change: no fusion, no averaging —
        just the one domain expert that matches the input's domain.
        """
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}'. "
                             f"Available: {self.domains}")
        cls    = self._encode(input_ids, attention_mask)
        routed = self.adapters[domain](cls)
        return self.heads[domain](routed)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# ============================================================
# 2.  Domain-specific teacher models
# ============================================================

# One best-fit model per domain.
# Chosen based on what each model was trained on.
DOMAIN_TEACHERS: dict[str, str] = {
    "finance":  "ProsusAI/finbert",                               # financial news
    "tech":     "cardiffnlp/twitter-roberta-base-sentiment-latest", # tech/startup news & tweets
    "business": "yiyanghkust/finbert-tone",                       # earnings & business tone
    "politics": "cardiffnlp/twitter-roberta-base-sentiment-latest", # news & political tweets
    "law":      "siebert/sentiment-roberta-large-english",         # strong general, handles legal prose
}
# Fallback for any domain not listed
DEFAULT_TEACHER = "siebert/sentiment-roberta-large-english"


class SingleTeacher:
    """
    Wraps one sentiment model. Handles any label vocabulary dynamically
    by reading id2label at load time — works for any pos/neg/neu model.
    """
    def __init__(self, model_name: str, device: str, temperature: float):
        self.device      = device
        self.temperature = temperature
        self.name        = model_name
        self.tokenizer   = AutoTokenizer.from_pretrained(model_name)
        self.model       = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(device)
        self.model.eval()

        # Map any label vocabulary → [0, 1] score
        label_to_score = {
            "positive": 1.0,  "negative": 0.0,  "neutral": 0.5,
            "pos":      1.0,  "neg":      0.0,  "neu":     0.5,
            "label_0":  0.0,  "label_1":  0.5,  "label_2": 1.0,
            "positive sentiment": 1.0, "negative sentiment": 0.0,
        }
        n = len(self.model.config.id2label)
        self.score_map = torch.zeros(n)
        for idx, lbl in self.model.config.id2label.items():
            self.score_map[int(idx)] = label_to_score.get(lbl.lower().strip(), 0.5)

    @torch.no_grad()
    def predict(self, texts: list[str]) -> dict:
        enc    = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(self.device)
        logits = self.model(**enc).logits.cpu()
        probs  = torch.softmax(logits, dim=-1)
        soft   = torch.softmax(logits / self.temperature, dim=-1)
        score  = (probs * self.score_map).sum(-1)

        # ── Normalize soft_probs to always be (B, 3) ─────────────────────
        # Some teachers output 2 classes (pos/neg only, no neutral).
        # The KL loss expects (B, 3) so we standardise here by mapping
        # every teacher's output to [neg_prob, neu_prob, pos_prob].
        n_classes = logits.shape[-1]
        if n_classes == 3:
            soft3 = soft   # already correct
        elif n_classes == 2:
            # Find which index is positive and which is negative
            # by checking score_map (1.0 = pos, 0.0 = neg)
            pos_idx = int((self.score_map == 1.0).nonzero(as_tuple=True)[0][0])
            neg_idx = 1 - pos_idx
            pos_p   = soft[:, pos_idx]
            neg_p   = soft[:, neg_idx]
            neu_p   = torch.zeros_like(pos_p)   # no neutral class → 0
            soft3   = torch.stack([neg_p, neu_p, pos_p], dim=1)  # (B, 3)
        else:
            # >3 classes: collapse to neg/neu/pos using score_map buckets
            neg_p = probs[:, self.score_map == 0.0].sum(-1)
            pos_p = probs[:, self.score_map == 1.0].sum(-1)
            neu_p = probs[:, self.score_map == 0.5].sum(-1)
            total = (neg_p + neu_p + pos_p).clamp(min=1e-9)
            soft3 = torch.stack([neg_p/total, neu_p/total, pos_p/total], dim=1)

        # hard_idx also mapped to 3-class space (0=neg, 1=neu, 2=pos)
        hard_score = score
        hard_idx3  = (hard_score > 0.6).long() * 2 + \
                     ((hard_score >= 0.4) & (hard_score <= 0.6)).long()
        # → 0 if bearish (<0.4), 1 if neutral (0.4-0.6), 2 if bullish (>0.6)

        return {
            "score":      score,                   # (B,) continuous [0,1]
            "hard_idx":   hard_idx3,               # (B,) always in {0,1,2}
            "soft_probs": soft3,                   # (B,3) always 3 classes
            "confidence": probs.max(-1).values,    # (B,) max prob
        }

    def unload(self):
        """Free GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DomainTeacherRegistry:
    """
    One teacher per domain, lazy-loaded and unloaded after use
    to keep memory footprint small.

      finance   → ProsusAI/finbert
      tech      → cardiffnlp/twitter-roberta-base-sentiment-latest
      business  → yiyanghkust/finbert-tone
      politics  → cardiffnlp/twitter-roberta-base-sentiment-latest
      law       → siebert/sentiment-roberta-large-english
    """

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._cache: dict[str, SingleTeacher] = {}

    def _get(self, domain: str) -> SingleTeacher:
        """Load teacher for domain if not already in cache."""
        if domain not in self._cache:
            model_name = DOMAIN_TEACHERS.get(domain, DEFAULT_TEACHER)
            print(f"[Teacher] '{domain}' → loading {model_name} …")
            try:
                self._cache[domain] = SingleTeacher(
                    model_name, self.cfg.device, self.cfg.temperature)
                print(f"           ✓ loaded")
            except Exception as e:
                print(f"           ✗ failed ({e}), using default teacher …")
                self._cache[domain] = SingleTeacher(
                    DEFAULT_TEACHER, self.cfg.device, self.cfg.temperature)
        return self._cache[domain]

    def label_batch(self, texts: list[str], domain: str) -> dict:
        """Label a batch using the domain-specific teacher."""
        teacher = self._get(domain)
        result  = teacher.predict(texts)

        # Keep only samples where teacher is confident enough
        keep = result["confidence"] >= self.cfg.min_teacher_confidence

        return {
            "hard_score": result["score"],
            "hard_idx":   result["hard_idx"],
            "soft_probs": result["soft_probs"],
            "keep":       keep,
        }

    def score_texts(self, texts: list[str], domain: str) -> list[float]:
        return self._get(domain).predict(texts)["score"].tolist()

    def unload(self, domain: str):
        """Free teacher memory after a domain is fully labelled."""
        if domain in self._cache:
            print(f"[Teacher] Unloading '{domain}' teacher …")
            self._cache[domain].unload()
            del self._cache[domain]


# ============================================================
# 3.  Dataset  (domain-aware)
# ============================================================

class DomainSentimentDataset(Dataset):
    """
    Each sample carries its domain label so the training loop can
    route it to the correct adapter + head.
    """
    def __init__(self, texts, domains, hard_scores, hard_idxs,
                 soft_probs, tokenizer, max_length, domain_to_idx):
        self.texts        = texts
        self.domains      = domains          # list of str
        self.hard_scores  = hard_scores
        self.hard_idxs    = hard_idxs
        self.soft_probs   = soft_probs
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.domain_to_idx = domain_to_idx  # str → int (for batching stats)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_length,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "domain":         self.domains[idx],            # str — used for routing
            "hard_score":     self.hard_scores[idx],
            "hard_idx":       self.hard_idxs[idx],
            "soft_probs":     self.soft_probs[idx],
        }


# ============================================================
# 4.  Loss  (same anti-collapse hybrid loss)
# ============================================================

class HybridLoss(nn.Module):
    def __init__(self, soft_weight: float = 0.5, temperature: float = 3.0,
                 margin: float = 0.15, rank_weight: float = 0.4):
        super().__init__()
        self.sw = soft_weight; self.T = temperature
        self.margin = margin;  self.rank_weight = rank_weight
        self.huber = nn.HuberLoss(delta=0.3)
        self.ce    = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.kl    = nn.KLDivLoss(reduction="batchmean")

    def _ranking_loss(self, score, target):
        si, sj = score.unsqueeze(1), score.unsqueeze(0)
        ti, tj = target.unsqueeze(1), target.unsqueeze(0)
        diff_t = ti - tj; diff_s = si - sj
        strong = diff_t.abs() > self.margin
        hinge  = torch.clamp(self.margin - diff_t.sign() * diff_s, min=0.0)
        return (hinge * strong.float()).sum() / strong.float().sum().clamp(min=1)

    def forward(self, score, logits, hard_score, hard_idx, soft_probs):
        huber = self.huber(score, hard_score)
        rank  = self._ranking_loss(score, hard_score)
        ce    = self.ce(logits, hard_idx)
        kl    = self.kl(torch.log_softmax(logits / self.T, -1), soft_probs) * self.T**2
        total = 0.25*huber + self.rank_weight*rank + (1-self.sw)*0.35*ce + self.sw*0.4*kl
        return total, {"huber": huber.item(), "rank": rank.item(),
                       "ce": ce.item(), "kl": kl.item()}


# ============================================================
# 5.  Data loading
# ============================================================

_TEXT_COLS = ["sentence","text","headline","title","content",
              "input","news","tweet","body","description"]

def _pick_col(columns) -> str:
    lc = {c.lower(): c for c in columns}
    for c in _TEXT_COLS:
        if c in lc: return lc[c]
    return list(columns)[0]

def _load_file(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delim))
        col = _pick_col(rows[0].keys())
        return [r[col] for r in rows if r.get(col,"").strip()]
    if ext == ".jsonl":
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        col  = _pick_col(rows[0].keys())
        return [str(r[col]) for r in rows if str(r.get(col,"")).strip()]
    if ext == ".json":
        data = json.load(open(path, encoding="utf-8"))
        rows = data if isinstance(data, list) else \
               data.get("train") or next(iter(data.values()))
        col  = _pick_col(rows[0].keys())
        return [str(r[col]) for r in rows if str(r.get(col,"")).strip()]
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df[_pick_col(df.columns)].dropna().astype(str).tolist()
    raise ValueError(f"Unsupported extension: {ext}")


def load_domain_texts(cfg: Config) -> dict[str, list[str]]:
    """Returns {domain: [text, ...]} capped at max_samples_per_domain."""
    domain_texts: dict[str, list[str]] = {d: [] for d in cfg.domains}

    for path, domain in cfg.data_files:
        if domain not in cfg.domains:
            print(f"[Data] SKIP {path} — unknown domain '{domain}'. "
                  f"Add it to cfg.domains first.")
            continue
        try:
            batch = _load_file(path)
            domain_texts[domain] += batch
            print(f"[Data] {domain:<12}  {path}  →  {len(batch):,} samples")
        except FileNotFoundError:
            print(f"[Data] SKIP {path} — file not found")
        except Exception as e:
            print(f"[Data] SKIP {path} — {e}")

    # Dedup, min-word filter, cap per domain
    for d in cfg.domains:
        texts = list(dict.fromkeys(
            t for t in domain_texts[d] if len(t.split()) >= 3
        ))
        random.shuffle(texts)
        domain_texts[d] = texts[: cfg.max_samples_per_domain]
        print(f"[Data] {d:<12}  → {len(domain_texts[d]):,} unique samples kept")

    total = sum(len(v) for v in domain_texts.values())
    if total == 0:
        raise RuntimeError("No samples loaded. Check cfg.data_files.")
    print(f"[Data] Total: {total:,} samples across {len(cfg.domains)} domains\n")
    return domain_texts


# ============================================================
# 6.  Trainer
# ============================================================

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.tok      = AutoTokenizer.from_pretrained(cfg.student_model)
        self.teachers = DomainTeacherRegistry(cfg)   # lazy-loads per domain
        self.model    = MultiDomainModel(cfg).to(cfg.device)
        self.loss_fn  = HybridLoss(cfg.soft_label_weight, cfg.temperature)
        self.domain_to_idx = {d: i for i, d in enumerate(cfg.domains)}

    # ----------------------------------------------------------
    def _label_corpus(self, domain_texts: dict[str, list[str]]) -> DomainSentimentDataset:
        all_texts, all_domains = [], []
        all_hs, all_hi, all_sp = [], [], []
        bs = 32

        for domain, texts in domain_texts.items():
            if not texts:
                print(f"[Label] {domain} — no samples, skipping")
                continue

            print(f"\n[Label] '{domain}' — {len(texts):,} samples")
            kept = 0

            for i in tqdm(range(0, len(texts), bs),
                          desc=f"  labelling {domain}", leave=False):
                batch  = texts[i: i+bs]
                # ← Domain-specific teachers used here
                labels = self.teachers.label_batch(batch, domain)
                keep   = labels["keep"]

                for text, k in zip(batch, keep.tolist()):
                    if k:
                        all_texts.append(text)
                        all_domains.append(domain)
                        kept += 1

                all_hs.append(labels["hard_score"][keep])
                all_hi.append(labels["hard_idx"][keep])
                all_sp.append(labels["soft_probs"][keep])

            dropped = len(texts) - kept
            print(f"         kept {kept:,} / {len(texts):,} "
                  f"({100*dropped/max(len(texts),1):.1f}% filtered)")

            # Free teacher GPU memory before loading the next domain's teachers
            self.teachers.unload(domain)

        # Score stretching across all domains
        scores = torch.cat(all_hs)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-6:
            scores = (scores - s_min) / (s_max - s_min) * 0.90 + 0.05

        print(f"\n[Label] Total kept: {len(all_texts):,}  "
              f"score [{scores.min():.3f}, {scores.max():.3f}]  "
              f"mean={scores.mean():.3f}  std={scores.std():.3f}")

        return DomainSentimentDataset(
            all_texts, all_domains, scores,
            torch.cat(all_hi), torch.cat(all_sp),
            self.tok, self.cfg.max_length, self.domain_to_idx,
        )

    # ----------------------------------------------------------
    def _collate(self, batch):
        """Custom collate: stack tensors, keep domain as list of strings."""
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "domain":         [b["domain"]     for b in batch],   # list[str]
            "hard_score":     torch.stack([b["hard_score"]     for b in batch]),
            "hard_idx":       torch.stack([b["hard_idx"]       for b in batch]),
            "soft_probs":     torch.stack([b["soft_probs"]     for b in batch]),
        }

    def _make_loaders(self, dataset):
        n_val  = max(1, int(len(dataset) * self.cfg.val_split))
        train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])
        return (
            DataLoader(train_ds, batch_size=self.cfg.batch_size,
                       shuffle=True, drop_last=True, collate_fn=self._collate),
            DataLoader(val_ds,   batch_size=self.cfg.batch_size,
                       shuffle=False, collate_fn=self._collate),
        )

    # ----------------------------------------------------------
    def _run_epoch(self, loader, optimizer, scheduler=None, train=True):
        """
        Key change: each batch is split by domain and routed to
        the correct adapter + head independently.
        """
        self.model.train(train)
        totals = {"total":0., "huber":0., "rank":0., "ce":0., "kl":0.}
        n = 0

        ctx = torch.no_grad() if not train else torch.enable_grad()
        with ctx:
            for batch in tqdm(loader, desc="train" if train else "val  ",
                              leave=False):
                ids  = batch["input_ids"].to(self.cfg.device)
                mask = batch["attention_mask"].to(self.cfg.device)
                hs   = batch["hard_score"].to(self.cfg.device)
                hi   = batch["hard_idx"].to(self.cfg.device)
                sp   = batch["soft_probs"].to(self.cfg.device)
                domains = batch["domain"]   # list[str]

                # ── Domain routing ────────────────────────────────────────
                # Group indices by domain, forward each group through its
                # own adapter+head, accumulate loss across domains.
                batch_loss = torch.tensor(0.0, device=self.cfg.device,
                                         requires_grad=train)
                batch_breakdown = {"huber":0.,"rank":0.,"ce":0.,"kl":0.}
                n_groups = 0

                unique_domains = list(dict.fromkeys(domains))  # preserve order
                for domain in unique_domains:
                    idx = torch.tensor(
                        [i for i, d in enumerate(domains) if d == domain],
                        device=self.cfg.device
                    )
                    if len(idx) < 2:   # need at least 2 for ranking loss
                        continue

                    score, logits = self.model(ids[idx], mask[idx], domain)
                    loss, breakdown = self.loss_fn(
                        score, logits, hs[idx], hi[idx], sp[idx]
                    )
                    batch_loss = batch_loss + loss
                    for k, v in breakdown.items():
                        batch_breakdown[k] += v
                    n_groups += 1

                if n_groups == 0:
                    continue

                batch_loss = batch_loss / n_groups   # average across domains

                if train:
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg.grad_clip)
                    optimizer.step()
                    if scheduler: scheduler.step()
                    optimizer.zero_grad()

                b = len(ids)
                totals["total"] += batch_loss.item() * b
                for k, v in batch_breakdown.items():
                    totals[k] += (v / n_groups) * b
                n += b

        return {k: v / max(n, 1) for k, v in totals.items()}

    # ----------------------------------------------------------
    def _make_optimizer(self, lr):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=self.cfg.weight_decay,
        )

    def _fmt(self, d: dict) -> str:
        return (f"total={d['total']:.4f} "
                f"[H={d['huber']:.3f} R={d['rank']:.3f} "
                f"CE={d['ce']:.3f} KL={d['kl']:.3f}]")

    # ----------------------------------------------------------
    def train(self):
        domain_texts = load_domain_texts(self.cfg)
        dataset      = self._label_corpus(domain_texts)
        train_loader, val_loader = self._make_loaders(dataset)

        steps   = lambda e: e * len(train_loader)
        warmup  = lambda s: int(s * self.cfg.warmup_ratio)
        best_val, patience = math.inf, self.cfg.early_stop_patience

        # ── Phase 1: adapters + heads only ───────────────────────────────
        print("\n" + "="*65)
        print(" Phase 1 — Training domain adapters (encoder frozen)")
        print("="*65)
        self.model.freeze_encoder()
        opt1   = self._make_optimizer(self.cfg.lr_frozen)
        sched1 = get_linear_schedule_with_warmup(
            opt1, warmup(steps(self.cfg.epochs_frozen)),
            steps(self.cfg.epochs_frozen))

        no_improve = 0
        for epoch in range(1, self.cfg.epochs_frozen + 1):
            tr  = self._run_epoch(train_loader, opt1, sched1, train=True)
            val = self._run_epoch(val_loader,   opt1, train=False)
            print(f"  Epoch {epoch}/{self.cfg.epochs_frozen}  "
                  f"train {self._fmt(tr)}  val={val['total']:.4f}")
            if val["total"] < best_val:
                best_val = val["total"]; no_improve = 0
                self._save("_phase1_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop (patience={patience})"); break

        # ── Phase 2: full fine-tune ───────────────────────────────────────
        print("\n" + "="*65)
        print(" Phase 2 — Full fine-tune (encoder unfrozen)")
        print("="*65)
        self.model.unfreeze_encoder()
        opt2   = self._make_optimizer(self.cfg.lr_unfrozen)
        sched2 = get_linear_schedule_with_warmup(
            opt2, warmup(steps(self.cfg.epochs_unfrozen)),
            steps(self.cfg.epochs_unfrozen))

        no_improve = 0
        for epoch in range(1, self.cfg.epochs_unfrozen + 1):
            tr  = self._run_epoch(train_loader, opt2, sched2, train=True)
            val = self._run_epoch(val_loader,   opt2, train=False)
            print(f"  Epoch {epoch}/{self.cfg.epochs_unfrozen}  "
                  f"train {self._fmt(tr)}  val={val['total']:.4f}")
            if val["total"] < best_val:
                best_val = val["total"]; no_improve = 0
                self._save(self.cfg.save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop (patience={patience})"); break

        print(f"\n✅  Best val loss: {best_val:.4f}")
        print(f"    Model saved → {self.cfg.save_path}")
        return self.model

    def _save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "config_dict": dataclasses.asdict(self.cfg),
        }, path)


# ============================================================
# 7.  Inference pipeline
# ============================================================

class DomainSentimentPipeline:
    """
    High-level API for training and inference.

    Training
    --------
    pipeline = DomainSentimentPipeline(cfg)
    pipeline.train()

    Inference — YOU supply the domain
    ----------------------------------
    pipeline.predict("Apple antitrust case dismissed.", domain="law")
    pipeline.predict("Fed raises rates.",               domain="finance")
    pipeline.predict("Senate passes AI bill.",          domain="politics")
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg     = cfg or Config()
        self.trainer = Trainer(self.cfg)
        self.model   = None

    def train(self):
        self.model = self.trainer.train()
        self._run_eval()
        return self

    # ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, text: str, domain: str) -> dict:
        """
        Score a single text using the adapter trained for `domain`.
        domain must be one of cfg.domains (e.g. 'finance', 'tech', etc.)
        """
        if self.model is None:
            raise RuntimeError("Call .train() or .load() first.")
        self.model.eval()
        enc = self.trainer.tok(
            text, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_length,
        ).to(self.cfg.device)
        score, _ = self.model(**enc, domain=domain)
        s     = score.item()
        label = "Positive" if s > 0.55 else ("Negative" if s < 0.35 else "Neutral")
        bar   = "█" * int(s * 30) + "░" * (30 - int(s * 30))
        print(f"  [{bar}] {s:.4f}  →  {label}  [{domain}]")
        print(f"  {text}")
        return {"score": round(s, 4), "label": label, "domain": domain}

    def predict_batch(self, items: list[tuple[str, str]]) -> list[dict]:
        """
        items: list of (text, domain) tuples
        Returns list of result dicts.
        """
        return [self.predict(text, domain) for text, domain in items]

    # ----------------------------------------------------------
    def _run_eval(self):
        eval_items = [
            ("Fed raises interest rates to combat inflation.",         "finance"),
            ("Quarterly earnings surpass Wall Street estimates.",       "finance"),
            ("Apple antitrust lawsuit dismissed by federal court.",     "law"),
            ("Supreme Court rules against patent infringement claim.",  "law"),
            ("Senate passes sweeping AI regulation bill.",             "politics"),
            ("President signs new trade tariff executive order.",      "politics"),
            ("Nvidia unveils next-gen GPU with 2x performance gains.", "tech"),
            ("Major data breach exposes 50 million user records.",     "tech"),
            ("Company announces record revenue and dividend hike.",    "business"),
            ("Factory output falls for third consecutive quarter.",    "business"),
        ]
        print("\n" + "="*72)
        print(f"  {'TEXT':<48} {'DOM':<10} {'TEACHER':>7} {'STUDENT':>7}")
        print("="*72)

        # Group by domain to batch teacher calls efficiently
        from collections import defaultdict
        by_domain: dict[str, list[int]] = defaultdict(list)
        for i, (_, domain) in enumerate(eval_items):
            by_domain[domain].append(i)

        teacher_scores = [None] * len(eval_items)
        for domain, idxs in by_domain.items():
            texts = [eval_items[i][0] for i in idxs]
            scores = self.trainer.teachers.score_texts(texts, domain)
            for i, s in zip(idxs, scores):
                teacher_scores[i] = s

        for i, (text, domain) in enumerate(eval_items):
            result = self.predict(text, domain)
            ts = teacher_scores[i]
            agree = "✓" if abs(ts - result["score"]) < 0.20 else "✗"
            print(f"  {text[:48]:<48} {domain:<10} {ts:>7.3f} {result['score']:>7.3f}  {agree}")

    # ----------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "DomainSentimentPipeline":
        """Load a saved model — no retraining, no teachers loaded."""
        print(f"[Load] Loading from {path} …")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg   = Config(**checkpoint["config_dict"])
        inst  = cls(cfg)
        inst.model = MultiDomainModel(cfg)
        inst.model.load_state_dict(checkpoint["model_state"])
        inst.model.eval()
        inst.model.to(cfg.device)
        print(f"[Load] Ready. Domains: {cfg.domains}")
        return inst


# ============================================================
# 8.  Entry point
# ============================================================

if __name__ == "__main__":
    cfg = Config(
        domains = ["finance", "tech", "business", "politics", "law"],

        data_files = [
            # Finance
            ("data/finance_phrasebank.jsonl",  "finance"),
            ("data/finance_twitter.jsonl",      "finance"),
            # Tech  — download from HuggingFace or use your own news files
            ("data/tech_news.jsonl",            "tech"),
            # Business
            ("data/business_news.jsonl",        "business"),
            # Politics
            ("data/politics_headlines.jsonl",   "politics"),
            # Law
            ("data/law_news.jsonl",             "law"),
        ],

        max_samples_per_domain = 3_000,
        epochs_frozen          = 5,
        epochs_unfrozen        = 3,
        batch_size             = 16,
        save_path              = "domain_sentiment_model.pt",
    )

    # ── Train ────────────────────────────────────────────────────────────
    pipeline = DomainSentimentPipeline(cfg)
    pipeline.train()

    # ── Or load a saved model and predict directly ────────────────────────
    # pipeline = DomainSentimentPipeline.load("domain_sentiment_model.pt")

    # ── Predict — you always specify the domain ───────────────────────────
    print("\n--- Inference examples ---")
    pipeline.predict("Fed unexpectedly cuts interest rates by 50bps.",   domain="finance")
    pipeline.predict("Massive accounting fraud uncovered at major bank.", domain="finance")
    pipeline.predict("Apple antitrust lawsuit dismissed.",                domain="law")
    pipeline.predict("Senate passes sweeping AI regulation bill.",        domain="politics")
    pipeline.predict("Nvidia unveils next-gen GPU, stock surges 12%.",    domain="tech")