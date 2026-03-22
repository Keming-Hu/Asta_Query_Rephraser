"""
Query Rephrasing with Staged Admission Mechanism
=================================================
Implements a bounded, staged query-admission mechanism for iterative
semantic query expansion, inspired by Asta's Paper Finder workflow.

Pipeline:
  1. Generate candidate rephrasings via LLM (two prompt styles from Asta)
  2. Admit/reject via staged mechanism:
       Stage 0 - Cosine safety net against original query
       Stage 1 - Cosine novelty against pivot queue
       Stage 2 - Softmax-free attention redundancy scoring
  3. Early-stop when convergence is detected

Author: [Your name]
"""

import os
import math
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# ANSI Color Helpers
# ─────────────────────────────────────────────

class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def green(s):  return f"{C.GREEN}{s}{C.RESET}"
def red(s):    return f"{C.RED}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def bold(s):   return f"{C.BOLD}{s}{C.RESET}"

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class Config:
    # Stage 0: cosine safety net against original query
    # reject if similarity to original < this value (drift prevention)
    tau_orig: float = 0.5

    # Stage 1: base cosine novelty threshold against pivot queue
    # reject if adaptive tau_pivot <= max cosine similarity to any pivot
    tau_pivot: float = 0.85

    # Stage 1: adaptive tau_pivot parameters
    # EMA smoothing factor for mean pairwise similarity
    ema_beta: float = 0.9
    # scaling factor for adaptive threshold adjustment
    tau_alpha: float = 0.15

    # Stage 1: tau_pivot stabilization parameters
    # max candidates in stabilization run before forcing exit
    tau_stability_max_candidates: int = 25
    # convergence band: last 5 tau values must all fall within avg +- this
    tau_stability_band: float = 0.02
    # how many last tau_pivot values to average for final stable value
    tau_stability_avg_window: int = 5

    # Stage 1: sliding window size for freeze detection
    stage1_window_size: int = 20

    # Stage 1: freeze queue when rejection proportion >= this value
    stage1_freeze_proportion: float = 0.70

    # Stage 2: stop when rejection proportion >= this value
    # sliding window size = len(queue), dynamic
    stage2_stop_proportion: float = 0.80

    # Accepted queries: cosine similarity upper limit between any pair
    # candidates too similar to existing accepted queries are not recorded
    tau_accept: float = 0.90

    # Hard upper limit on total candidate queries generated
    max_candidates: int = 150

    # Maximum pivot queue size
    max_queue_size: int = 20

    # Embedding model (local, CPU-friendly)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # OpenAI model for LLM generation
    openai_model: str = "gpt-4o-mini"


# ─────────────────────────────────────────────
# Embedding Utilities
# ─────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps a HuggingFace transformer to provide:
      - pooled sentence embeddings (for cosine similarity)
      - token-level hidden states (for attention scoring)
    """

    def __init__(self, model_name: str):
        print(f"[Init] Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print(f"[Init] Model loaded successfully.\n")

    def encode(self, text: str) -> tuple[np.ndarray, torch.Tensor]:
        """
        Returns:
          pooled_embedding: np.ndarray of shape (d,)  — for cosine similarity
          token_hidden_states: torch.Tensor (seq_len, d) — for attention scoring
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Token-level hidden states: shape (1, seq_len, d)
        token_hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, d)

        # Mean pool for sentence-level embedding
        attention_mask = inputs["attention_mask"].squeeze(0)  # (seq_len,)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (seq_len, 1)
        sum_hidden = (token_hidden_states * mask_expanded).sum(0)
        pooled = sum_hidden / mask_expanded.sum(0).clamp(min=1e-9)
        pooled_np = F.normalize(pooled, dim=0).numpy()

        return pooled_np, token_hidden_states


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def is_accepted_query_novel(
    candidate_embedding: np.ndarray,
    accepted_queries_embeddings: list[np.ndarray],
    tau_accept: float,
) -> bool:
    """
    Returns True if candidate is novel enough to be added to accepted_queries.
    Checks cosine similarity against all already-accepted query embeddings.
    Rejects if any pair exceeds tau_accept.
    """
    for existing_emb in accepted_queries_embeddings:
        if cosine_similarity(candidate_embedding, existing_emb) >= tau_accept:
            return False
    return True


def compute_mean_pairwise_similarity(queue: "PivotQueue") -> float:
    """Compute mean pairwise cosine similarity among all pivots in the queue."""
    n = len(queue)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_similarity(queue.pivots[i].embedding, queue.pivots[j].embedding)
            count += 1
    return total / count


def compute_adaptive_tau_pivot(
    queue: "PivotQueue",
    config: "Config",
    ema_state: dict,
) -> float:
    """
    Compute adaptive tau_pivot using EMA + bias correction.

    ema_state is a mutable dict with keys:
      'ema'  : current EMA value (float)
      'step' : number of EMA updates so far (int)

    No warmup — EMA runs from the very first call.
    Early oscillation naturally prevents premature stabilization.
    """
    s_t = compute_mean_pairwise_similarity(queue)
    ema_state["ema"] = config.ema_beta * ema_state["ema"] + (1 - config.ema_beta) * s_t
    ema_state["step"] += 1

    # Bias correction
    s_hat = ema_state["ema"] / (1 - config.ema_beta ** ema_state["step"])

    # Adaptive threshold: tighten when pivots are dense, relax when diverse
    tau_adaptive = config.tau_pivot - config.tau_alpha * (1 - s_hat)
    return float(np.clip(tau_adaptive, 0.5, 0.99))


# ─────────────────────────────────────────────
# Softmax-free Attention Redundancy Score
# ─────────────────────────────────────────────

def attention_redundancy_score(
    X_A: torch.Tensor,  # pivot A token hidden states: (p, d)
    X_B: torch.Tensor,  # candidate B token hidden states: (q, d)
) -> float:
    """
    Computes how strongly pivot A can semantically account for candidate B.

    Using identity projections (W_Q = W_K = W_V = I) for simplicity.

    Steps:
      S_AB = X_A @ X_B^T                    shape: (p, q)
      S_AB_norm = S_AB / (q * sqrt(d))      normalized
      H_AB = S_AB_norm @ X_B                shape: (p, d)
      score = mean over rows of ||H_AB[i,:]||_2

    Larger score => B is more redundant w.r.t. A
    """
    p, d = X_A.shape
    q = X_B.shape[0]

    # Raw interaction matrix
    S_AB = torch.matmul(X_A, X_B.T)  # (p, q)

    # Normalize
    S_AB_norm = S_AB / (q * math.sqrt(d))  # (p, q)

    # Value aggregation
    H_AB = torch.matmul(S_AB_norm, X_B)  # (p, d)

    # Row norms, then average
    row_norms = torch.norm(H_AB, dim=1)  # (p,)
    score = row_norms.mean().item()

    return score


# ─────────────────────────────────────────────
# Pivot Queue
# ─────────────────────────────────────────────

@dataclass
class Pivot:
    text: str
    embedding: np.ndarray          # pooled embedding for cosine
    token_states: torch.Tensor     # token-level for attention
    key: float = 0.0               # avg attention redundancy w.r.t. other pivots


class PivotQueue:
    """
    Fixed-size bounded memory of representative queries (semantic frontier).
    Maintains at most max_size pivots.
    Each pivot has a 'key' = average attention redundancy against other pivots.
    High key = redundant pivot. Low key = frontier-defining pivot.
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.pivots: list[Pivot] = []

    def __len__(self) -> int:
        return len(self.pivots)

    def is_full(self) -> bool:
        return len(self.pivots) >= self.max_size

    def add(self, pivot: Pivot) -> None:
        """Add a pivot and recompute all keys."""
        self.pivots.append(pivot)
        self._recompute_keys()

    def replace_worst(self, new_pivot: Pivot) -> str:
        """Replace the pivot with highest key (most redundant) with new_pivot."""
        worst_idx = max(range(len(self.pivots)), key=lambda i: self.pivots[i].key)
        replaced_text = self.pivots[worst_idx].text
        self.pivots[worst_idx] = new_pivot
        self._recompute_keys()
        return replaced_text

    def get_worst_key(self) -> float:
        """Return the highest key value (most redundant pivot)."""
        if not self.pivots:
            return 0.0
        return max(p.key for p in self.pivots)

    def _recompute_keys(self) -> None:
        """Recompute attention redundancy key for each pivot."""
        n = len(self.pivots)
        if n < 2:
            for p in self.pivots:
                p.key = 0.0
            return

        for i, pivot_i in enumerate(self.pivots):
            scores = []
            for j, pivot_j in enumerate(self.pivots):
                if i != j:
                    # key(p_i) = avg s(p_j, p_i) over j != i
                    # i.e., how strongly other pivots account for p_i
                    s = attention_redundancy_score(pivot_j.token_states, pivot_i.token_states)
                    scores.append(s)
            pivot_i.key = float(np.mean(scores))

    def max_cosine_similarity(self, embedding: np.ndarray) -> float:
        """Max cosine similarity between embedding and any pivot."""
        if not self.pivots:
            return 0.0
        return max(cosine_similarity(embedding, p.embedding) for p in self.pivots)

    def avg_attention_redundancy(self, X_B: torch.Tensor) -> float:
        """Average attention redundancy of candidate B against all pivots."""
        if not self.pivots:
            return 0.0
        scores = [attention_redundancy_score(p.token_states, X_B) for p in self.pivots]
        return float(np.mean(scores))

    def texts(self) -> list[str]:
        return [p.text for p in self.pivots]


# ─────────────────────────────────────────────
# LLM Query Generation (Asta's two prompt styles)
# ─────────────────────────────────────────────

PROMPT_STYLE_1 = """
Given the following search query, formulate {max_output} different natural language queries to run on a dense retrieval index to find papers that match the original search query.
The dense retrieval index contains research papers taken from the arXiv and ACL Anthology.
Be creative, and try to formulate queries that are different from each other.
Each passage in the dense index is a single sentence.
Drop phrases like "a paper about..." or "studies showing...".
- For example for the original search query "papers about efficient language modeling", a good query could be "methods for efficient language modeling", but a bad query would be "papers that talk about efficient language modeling".
DO NOT include general preferences such as "recent paper", "highly-cited paper", as these are not expected to be found in the text of the paper.
The index does not support logical operators like "AND", "OR", "NOT", "-", "+", "&" etc.

Original Search Query: ```{search_query}```

Return ONLY a JSON object with key "alternative_queries" containing a list of query strings.
"""

PROMPT_STYLE_2 = """
Your task is to come up with up to {max_output} alternative search queries that will help find passages that answer the following search query.
The queries will be run on a dense index that contains passages from academic research papers.
I am NOT looking for simple synonym paraphrases of common words, as these are captured by the index embeddings. Try using some reasoning to come up with interesting new ways to answer the original query.
Make sure you use wording that is actually used within the searched for domain. Don't just give arbitrary synonyms.
Drop phrases like "study about...", "research showing...", "evidence for...", as these are already true for all passages in the index of academic papers.

Examples:
Example: "wide transformer models":
BAD output: "expansive transformer models"
GOOD output: "shallow transformer models"
GOOD output: "wide attention-based models"

Search query: {search_query}

Return ONLY a JSON object with key "alternative_queries" containing a list of query strings.
"""


def generate_candidates(
    client: OpenAI,
    search_query: str,
    model: str,
    batch_size: int = 3,
    style: int = 1
) -> list[str]:
    """
    Generate a batch of candidate rephrasings using one of Asta's two prompt styles.
    Returns a list of query strings.
    """
    import json

    prompt_template = PROMPT_STYLE_1 if style == 1 else PROMPT_STYLE_2
    prompt = prompt_template.format(search_query=search_query, max_output=batch_size)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        candidates = data.get("alternative_queries", [])
        return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    except Exception as e:
        print(f"  [LLM Error] {e}")
        return []


# ─────────────────────────────────────────────
# Main Admission Loop
# ─────────────────────────────────────────────

def run_query_expansion(
    original_query: str,
    config: Config,
    openai_api_key: str,
) -> list[str]:
    """
    Main entry point. Runs the full staged query admission mechanism.

    Returns:
        List of accepted pivot query texts representing the semantic frontier.
    """

    print(bold("═" * 60))
    print(bold("  QUERY EXPANSION WITH STAGED ADMISSION MECHANISM"))
    print(bold("═" * 60))
    print(f"  Original query: {cyan(original_query)}")
    print(f"  tau_orig={config.tau_orig} | tau_pivot={config.tau_pivot} | tau_accept={config.tau_accept} | tau_band={config.tau_stability_band}")
    print(f"  stage1_freeze={config.stage1_freeze_proportion} | stage2_stop={config.stage2_stop_proportion} | max_candidates={config.max_candidates} | max_queue={config.max_queue_size}")
    print()

    # Initialize models and clients
    embed_model = EmbeddingModel(config.embedding_model)
    client = OpenAI(api_key=openai_api_key)

    # Encode original query
    orig_embedding, orig_token_states = embed_model.encode(original_query)

    # Initialize pivot queue with original query
    original_pivot = Pivot(
        text=original_query,
        embedding=orig_embedding,
        token_states=orig_token_states,
        key=0.0
    )
    queue = PivotQueue(max_size=config.max_queue_size)
    queue.add(original_pivot)
    print(f"[Queue] Initialized with original query.\n")

    # Tracking variables
    total_candidates = 0
    queue_frozen = False
    stage1_window: list[int] = []   # 1 = rejected, 0 = accepted (Stage 1 sliding window)
    stage2_window: list[int] = []   # 1 = rejected, 0 = accepted (Stage 2 sliding window)
    prompt_style = 1                # alternate between style 1 and 2
    ema_state = {"ema": 0.0, "step": 0}  # EMA state for adaptive tau_pivot

    # Tracking variables
    total_candidates = 0
    queue_frozen = False
    stage1_window: list[int] = []
    stage2_window: list[int] = []
    prompt_style = 1
    ema_state = {"ema": 0.0, "step": 0}
    accepted_queries: list[str] = []  # all useful queries (Stage 1 + Stage 2 accepted)
    accepted_queries_embeddings: list[np.ndarray] = []  # embeddings for cosine dedup

    # Stabilization run tracking
    tau_pivot_history: list[float] = []
    cached_stage1: list[tuple] = []
    stabilization_candidates = 0
    tau_pivot_stable: float | None = None

    print(yellow("═" * 60))
    print(yellow("  PHASE 1 — STABILIZATION RUN (accept all, observe semantic space)"))
    print(yellow("═" * 60) + "\n")

    # ── Phase 1: Stabilization run ──
    while tau_pivot_stable is None:

        batch = generate_candidates(
            client=client,
            search_query=original_query,
            model=config.openai_model,
            batch_size=3,
            style=prompt_style
        )
        prompt_style = 2 if prompt_style == 1 else 1

        if not batch:
            print("  [Warning] Empty batch from LLM, skipping.")

        for candidate_text in batch:
            if stabilization_candidates >= config.tau_stability_max_candidates:
                break

            stabilization_candidates += 1
            total_candidates += 1
            print(cyan(f"[Stab #{stabilization_candidates}] \"{candidate_text}\""))

            cand_embedding, cand_token_states = embed_model.encode(candidate_text)
            sim_to_orig = cosine_similarity(orig_embedding, cand_embedding)
            print(f"  Stage 0 | sim_to_orig = {sim_to_orig:.4f} (threshold >= {config.tau_orig})")

            if sim_to_orig < config.tau_orig:
                print(red(f"  → REJECTED (Stage 0: semantic drift)"))
                continue

            # Cache all Stage 1 candidates
            cached_stage1.append((candidate_text, cand_embedding, cand_token_states))

            # Accept unconditionally — pure observation, no filtering
            queue.add(Pivot(
                text=candidate_text,
                embedding=cand_embedding,
                token_states=cand_token_states
            ))

            # Compute and record tau after every acceptance
            current_tau = compute_adaptive_tau_pivot(queue, config, ema_state)
            tau_pivot_history.append(current_tau)
            if len(tau_pivot_history) > config.tau_stability_avg_window:
                tau_pivot_history.pop(0)

            print(green(f"  → Accepted (queue now {len(queue)}), tau_pivot = {current_tau:.4f}"))

            # Check Exit 1: last 5 tau values within avg +- band
            if len(tau_pivot_history) == config.tau_stability_avg_window:
                avg = sum(tau_pivot_history) / len(tau_pivot_history)
                if all(abs(v - avg) <= config.tau_stability_band for v in tau_pivot_history):
                    tau_pivot_stable = avg
                    print(yellow(f"\n[Stabilized] tau_pivot converged to {tau_pivot_stable:.4f} "
                          f"(avg of last {config.tau_stability_avg_window}, all within ±{config.tau_stability_band})\n"))
                    break

        # Check Exit 2: hard cap hit
        if stabilization_candidates >= config.tau_stability_max_candidates and tau_pivot_stable is None:
            if tau_pivot_history:
                tau_pivot_stable = sum(tau_pivot_history) / len(tau_pivot_history)
            else:
                tau_pivot_stable = config.tau_pivot
            print(yellow(f"\n[Stabilized] Hard cap hit. Using avg of last {len(tau_pivot_history)} "
                  f"tau values = {tau_pivot_stable:.4f}\n"))

    print(yellow("═" * 60))
    print(yellow(f"  PHASE 2 — RETROACTIVE RE-EVALUATION ({len(cached_stage1)} cached candidates)"))
    print(yellow("═" * 60) + "\n")

    queue = PivotQueue(max_size=config.max_queue_size)
    queue.add(original_pivot)
    stage1_window = []

    for candidate_text, cand_embedding, cand_token_states in cached_stage1:
        max_sim = queue.max_cosine_similarity(cand_embedding)
        print(cyan(f"[Re-eval] \"{candidate_text}\""))
        print(f"  max_sim = {max_sim:.4f} | stable tau_pivot = {tau_pivot_stable:.4f}")

        if max_sim < tau_pivot_stable:
            queue.add(Pivot(
                text=candidate_text,
                embedding=cand_embedding,
                token_states=cand_token_states
            ))
            accepted_queries.append(candidate_text)
            accepted_queries_embeddings.append(cand_embedding)
            stage1_window.append(0)
            print(green(f"  → ACCEPTED (queue now {len(queue)})"))
        else:
            stage1_window.append(1)
            print(red(f"  → REJECTED"))

        if len(stage1_window) > config.stage1_window_size:
            stage1_window.pop(0)

    print(yellow(f"\n[Re-eval complete] Queue size: {len(queue)}\n"))

    if len(queue) >= config.max_queue_size:
        queue_frozen = True
        print(yellow(f"[Queue Frozen] Max size reached after re-evaluation. Transitioning to Stage 2.\n"))
    elif len(stage1_window) == config.stage1_window_size:
        rejection_proportion = sum(stage1_window) / config.stage1_window_size
        if rejection_proportion >= config.stage1_freeze_proportion:
            queue_frozen = True
            print(yellow(f"[Queue Frozen] Re-eval rejection proportion {rejection_proportion:.2f} "
                  f">= {config.stage1_freeze_proportion}. Transitioning to Stage 2.\n"))

    # ── Phase 3: Continue generation ──
    print(yellow("═" * 60))
    print(yellow(f"  PHASE 3 — CONTINUE GENERATION (queue_frozen={queue_frozen})"))
    print(yellow("═" * 60) + "\n")

    while total_candidates < config.max_candidates:

        batch = generate_candidates(
            client=client,
            search_query=original_query,
            model=config.openai_model,
            batch_size=3,
            style=prompt_style
        )
        prompt_style = 2 if prompt_style == 1 else 1

        if not batch:
            print("  [Warning] Empty batch from LLM, skipping.")

        for candidate_text in batch:
            if total_candidates >= config.max_candidates:
                print(yellow(f"\n[Stop] Hard upper limit reached ({config.max_candidates} candidates)."))
                break

            total_candidates += 1
            print(cyan(f"[Candidate #{total_candidates}] \"{candidate_text}\""))

            cand_embedding, cand_token_states = embed_model.encode(candidate_text)

            # Stage 0
            sim_to_orig = cosine_similarity(orig_embedding, cand_embedding)
            print(f"  Stage 0 | sim_to_orig = {sim_to_orig:.4f} (threshold >= {config.tau_orig})")
            if sim_to_orig < config.tau_orig:
                print(red(f"  → REJECTED [Stage 0] (semantic drift)"))
                continue

            # Stage 1 (stable tau_pivot)
            max_sim_to_queue = queue.max_cosine_similarity(cand_embedding)
            print(f"  Stage 1 | max_sim = {max_sim_to_queue:.4f} | tau = {tau_pivot_stable:.4f}")

            if not queue_frozen:
                if max_sim_to_queue >= tau_pivot_stable:
                    print(red(f"  → REJECTED [Stage 1] (not novel enough)"))
                    stage1_window.append(1)
                else:
                    queue.add(Pivot(
                        text=candidate_text,
                        embedding=cand_embedding,
                        token_states=cand_token_states
                    ))
                    accepted_queries.append(candidate_text)
                    accepted_queries_embeddings.append(cand_embedding)
                    stage1_window.append(0)
                    print(green(f"  → ACCEPTED [Stage 1] (queue now {len(queue)}/{config.max_queue_size})"))

                    if len(queue) >= config.max_queue_size:
                        queue_frozen = True
                        print(yellow(f"\n[Queue Frozen] Max size reached. Transitioning to Stage 2.\n"))

                if len(stage1_window) > config.stage1_window_size:
                    stage1_window.pop(0)

                if len(stage1_window) == config.stage1_window_size and not queue_frozen:
                    rejection_proportion = sum(stage1_window) / config.stage1_window_size
                    print(f"  Stage 1 | window rejection proportion = {rejection_proportion:.2f} "
                          f"(freeze threshold >= {config.stage1_freeze_proportion})")
                    if rejection_proportion >= config.stage1_freeze_proportion:
                        queue_frozen = True
                        print(yellow(f"\n[Queue Frozen] Rejection proportion {rejection_proportion:.2f} "
                              f">= {config.stage1_freeze_proportion}. "
                              f"Queue size fixed at {len(queue)}. Transitioning to Stage 2.\n"))

            else:
                # Stage 2
                avg_red = queue.avg_attention_redundancy(cand_token_states)
                worst_key = queue.get_worst_key()
                window_size = len(queue)
                print(f"  Stage 2 | avgRed = {avg_red:.4f} | worst_key = {worst_key:.4f} | window = {window_size}")

                if avg_red < worst_key:
                    # Layer 1: attention passed — now check tau_accept
                    if is_accepted_query_novel(cand_embedding, accepted_queries_embeddings, config.tau_accept):
                        # Both checks passed — record in answer set
                        accepted_queries.append(candidate_text)
                        accepted_queries_embeddings.append(cand_embedding)
                        stage2_window.append(0)

                        # Layer 2: cosine novelty check for pivot replacement
                        max_sim_to_queue = queue.max_cosine_similarity(cand_embedding)
                        if max_sim_to_queue < tau_pivot_stable:
                            replaced = queue.replace_worst(
                                Pivot(text=candidate_text, embedding=cand_embedding,
                                      token_states=cand_token_states)
                            )
                            print(green(f"  → ACCEPTED [recorded + replaced \"{replaced[:40]}...\"]"))
                        else:
                            print(green(f"  → ACCEPTED [recorded only] (cosine={max_sim_to_queue:.4f} >= tau, no replacement)"))
                    else:
                        # Failed tau_accept — rejected immediately
                        stage2_window.append(1)
                        print(red(f"  → REJECTED [tau_accept] (too similar to existing accepted query)"))
                else:
                    stage2_window.append(1)
                    print(red(f"  → REJECTED [attention] (avgRed={avg_red:.4f} >= worst_key={worst_key:.4f})"))

                if len(stage2_window) > window_size:
                    stage2_window.pop(0)

                if len(stage2_window) == window_size:
                    rejection_proportion = sum(stage2_window) / window_size
                    print(f"  Stage 2 | window rejection proportion = {rejection_proportion:.2f} "
                          f"(stop threshold >= {config.stage2_stop_proportion})")
                    if rejection_proportion >= config.stage2_stop_proportion:
                        print(yellow(f"\n[Stop] Early stop: Stage 2 rejection proportion "
                              f"{rejection_proportion:.2f} >= {config.stage2_stop_proportion}."))
                        _print_final_results(queue, accepted_queries, total_candidates)
                        return accepted_queries

    _print_final_results(queue, accepted_queries, total_candidates)
    return accepted_queries


def _print_final_results(queue: PivotQueue, accepted_queries: list[str], total_candidates: int):
    """Print the final accepted queries and pivot queue."""
    print("\n" + bold("═" * 60))
    print(bold("  FINAL ACCEPTED QUERIES"))
    print(bold("═" * 60))
    print(f"  Total candidates generated: {total_candidates}")
    print(f"  Accepted queries: {green(str(len(accepted_queries)))} | Final pivot queue: {cyan(str(len(queue)))}\n")
    for i, text in enumerate(accepted_queries, 1):
        print(green(f"  {i:2d}. {text}"))
    print(bold("\n  — Final Pivot Queue —"))
    for i, text in enumerate(queue.texts(), 1):
        print(cyan(f"  {i:2d}. {text}"))
    print(bold("═" * 60))


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Set your OpenAI API key here ──
    #YOUR_API_KEY = os.environ.get("YOUR_API_KEY", "YOUR_API_KEY")

    # ── User input query ──
    QUERY = input("Enter your search query: ").strip()

    # ── Config (safety net disabled for initial observation: set max_candidates high) ──
    config = Config(
        tau_orig=0.5,
        tau_pivot=0.85,
        ema_beta=0.9,
        tau_alpha=0.15,
        tau_stability_max_candidates=25,
        tau_stability_band=0.002,
        tau_stability_avg_window=5,
        stage1_window_size=20,
        stage1_freeze_proportion=0.70,
        stage2_stop_proportion=0.80,
        tau_accept=0.90,
        max_candidates=150,
        max_queue_size=20,
    )

    accepted_queries = run_query_expansion(
        original_query=QUERY,
        config=config,
        openai_api_key=OPENAI_API_KEY,
    )