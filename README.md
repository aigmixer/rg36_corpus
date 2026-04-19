# RG 36 Corpus Pipeline

Corpus engineering pipeline for the **AU Financial Advice Detector** — a CAA-based axis that distinguishes General Information from Personal Financial Product Advice under ASIC RG 36 / Corporations Act 2001 s766B.

---

## Current Status

**MPV target:** Both passed — TPR ≥ 75% and TPR ≥ 95% at FPR = 0%.

**Deployed golden path (Entry 948, T1):** Single-gate mean-pool CAA at Layer 14 — TPR=97.5% at FPR=0% on 400-sample AU holdout (N=200 ADV, N=200 BEN). This is the **empirically exhausted floor for linear methods**. Entries 952–954 systematically tested all alternatives (benign augmentation, layer sweep L10–L14, SVM-L2 KV-Trace, atlas 2D-AND confound gate, L8 CAA, spherical K-Means K=2) — none improved on 97.5% TPR at 0% FPR. Non-linear methods are the next frontier.

**Axis origin (Entry 945, T1):** Mean-pool CAA from v8 corpus at Layer 14 — Cohen's d=5.51, TPR=98.0% at FPR=0% on 200-sample AU holdout. Both MPV targets passed. The axis in `m-dad/concepts/asic_rg36_au_coach_v1_meanpool/vectors/advice_intensity_axis.pt` is the production deployment vector.

**Extraction method:** `axis-audit compile ... --pooling mean` (default since v0.4). Mean-pooling over assistant response tokens at Layer 14 raised Cohen's d from 1.19 (last-token) to 5.51. This was the decisive improvement.

**Corpus generation:** Complete. `v942_au_coach_corpus_runner.py` generated 500 calibration + 200 holdout pairs grounded in AFCA breach patterns. Corpus files: `configs/au_rg36_coach_calibration_v1.json` (training) and `configs/au_rg36_coach_holdout_v1.json` (holdout, held out during training). A hedged-only filtered subset (`m-dad/configs/au_rg36_coach_hedged_only_v1.json`, 257 pairs) was derived for Gate 2 extraction experiments.

**Structural ceiling (n-1 flaw, documented Entry 953–954):** At every tested layer (L8–L14), the calibration corpus benign ceiling underestimates the holdout benign ceiling. This means any Gate 2 axis calibrated at 0-FPR on the calibration set will produce FPR>0% on holdout. The 5 FNs are geometrically entangled with a benign tail — they cannot be recovered by any linear separation at calibration-based thresholds.

**Previous failure history:**
- Entry 941 (last-token, v8 corpus): d=2.63, TPR=63.3% — near-threshold softening on conditional/SOA/email ADV patterns
- Entry 942 failure analysis: 22/60 ADV missed in two clusters (near-threshold softening N=19, format-shifted N=3)
- Root cause: last-token capture blind to distributed directive signal in long-format responses
- Earlier failure: Vocabulary overfitting — v5 CAA trained on US-only fund names (VTSMX, IEFA) failed completely on UK/ESG vocabulary. Fix was AU-specific corpus grounded in real AFCA breach determinations.

---

## Architecture

### The Legal Boundary

Personal advice (triggers compliance obligations) = a recommendation or opinion made to a specific person where the adviser has considered (or claims to consider) their personal circumstances.

General information = factual statements about products, markets, or regulations. No personal reference.

**Key linguistic marker:** second-person possessive + directive verb.
- PERSONAL: *"you should rollover your REST Super to an SMSF"*
- GENERAL: *"rolling over super to an SMSF is an option the relevant person may consider"*

### Why AFCA Is the Gold Standard

Purely synthetic templates (Strategy 2) are nearly orthogonal to real breach data (cosine ~0.07, Entry 242). The `afca_only_pairs.jsonl` dataset is derived from 148 binding AFCA determinations — real cases where licensees were found to have given personal advice without a licence. These are the only ground-truth pairs the CAA extractor should be trained on.

---

## Data Sources

| Source | Location | Content |
| :--- | :--- | :--- |
| AFCA determinations | `raw/afca/` | 148 PDFs — real AU advice breach cases (SMSF, REST Super, LRBA, NRAS) |
| ASIC regulatory guides | `raw/asic/` | RG 36, 175, 244, 274 + enforcement action PDFs |
| MoneySmart | `raw/moneysmart/` | Factual consumer finance education (HTML) |
| RBA | `raw/rba/` | Monetary policy minutes (factual economic data) |

---

## Corpus Files

### Active (use these)

| File | N | Format | Notes |
| :--- | ---: | :--- | :--- |
| `pairs/afca_only_pairs.jsonl` | 4,391 | Legacy (`vector_a`/`vector_b`) | **Gold standard.** Word-level substitution pairs from AFCA PDFs. CAA training data. |
| `pairs/rg36_pairs_v4_gold.jsonl` | 9,390 | Legacy | AFCA gold (4,391) + hybrid injection (4,999). Hybrid portion is lower quality — prefer afca_only for CAA. |
| `pairs/rg36_pairs.jsonl` | 10,000 | Legacy | Baseline unfiltered synthetic. Use only for scale augmentation. |
| `pairs/asic_new_candidates.jsonl` | 536 | Legacy | Exploratory ASIC extraction. Not validated. |
| `pairs/asic_gold_candidates.jsonl` | — | Legacy | ASIC candidate pairs, pre-audit. |
| `pairs/fscp_gold_repaired.jsonl` | 20 | Legacy | High-fidelity FSCP breach pairs. |
| `pairs/fscp_gold_candidates.jsonl` | — | Legacy | FSCP candidates, pre-audit. |
| `pairs/ahpra_gold_v245.jsonl` | 20 | Legacy | Cross-domain anchor (medical advice boundary). |
| `pairs/ahpra_medical_gold.jsonl` | — | Legacy | Medical advice gold pairs. |

**Coach-generated AU corpus** (output of `scripts/v942_au_coach_corpus_runner.py`):

| File | N | Format | Notes |
| :--- | ---: | :--- | :--- |
| `configs/au_rg36_coach_calibration_v1.json` | 500 | v8 (`target_text`/`noise_text`, chat-formatted) | CAA training set. Groq-generated, AFCA-grounded. |
| `configs/au_rg36_coach_holdout_v1.json` | 200 | v8 | Independent holdout. Different seed. |

### Archived (do not use)

Files in `archived/` are broken Gemini implementations superseded by the current approach:

| File | Reason archived |
| :--- | :--- |
| `archived/bridge_rg36_to_v2.py` | v9 plan Phase 1 — injected preambles onto AFCA text. Produced grammatically valid but semantically incorrect pairs. Wrong approach. |
| `archived/generate_rg36_hybrid.py` | Local Llama 3.2-3B generation. Produced ASIC citation text with preambles prepended — not realistic conversation format. Garbage output. |
| `archived/rg36_pairs_contaminated_hybrid.jsonl` | 10,000-pair synthetic hybrid. Found geometrically divergent from real AFCA data (Entry 242, cosine ~0.07). Caused vocabulary overfitting in downstream models. |

---

## Active Scripts

### `pair_generator.py`
Template-based pair generation (Strategy 2). **27 templates** covering:
- Super: SG rate, consolidation, concessional contributions, SMSF, LRBA
- Mortgage: variable/fixed rate, offset account
- Investment: ASX ETF (VAS, BetaShares A200), diversification, DCA
- Insurance: life, income protection, TPD
- Debt/budget: credit card, emergency fund
- **All 4 Entry 942 failure mode patterns:** conditional/softened, SOA letter format, email directive, disclaimer-hedged

Variable pools include AU-specific fund names: AustralianSuper, Hostplus, REST Super, CBUS, UniSuper, AMP Flexible Super.

### `scripts/v942_au_coach_corpus_runner.py`
Model-agnostic coach corpus generator. Uses Groq (llama-3.3-70b-versatile) to generate matched ADV/BEN conversation pairs grounded in real AFCA breach patterns. Outputs v8-compatible corpus JSON ready for CAA extraction.

```bash
# Dry run (API check only)
GROQ_API_KEY=<key> python3 scripts/v942_au_coach_corpus_runner.py --dry-run

# Full run
GROQ_API_KEY=<key> python3 scripts/v942_au_coach_corpus_runner.py \
  --n-calibration 500 --n-holdout 200
```

### `ingestion_pipeline.py`
PDF/HTML ingestion → `ParsedDocument`. Handles ASIC and AFCA layouts.

### `pair_auditor.py`
Quality control: statutory marker verification, NLI consistency checks, chat template validation.

### `run_pipeline.py`
Orchestrates ingestion → generation → audit.

```bash
python3 -m rg36_corpus.run_pipeline --target 1000
python3 -m rg36_corpus.run_pipeline --demo   # 10 pairs, end-to-end smoke test
```

---

## Data Schema

### Legacy format (`afca_only_pairs.jsonl`, `rg36_pairs_v4_gold.jsonl`)
```json
{
  "pair_id": "uuid",
  "vector_a": "General information text",
  "label_a": "GENERAL",
  "vector_b": "Personal advice text",
  "label_b": "PERSONAL",
  "has_second_person_possessive": true,
  "has_westpac_pattern": false,
  "source_doc": "784656.pdf",
  "metadata": {"violations": ["your needs"]}
}
```

### v8 format (`au_rg36_coach_calibration_v1.json`)
```json
{
  "pair_id": "uuid",
  "target_axis": "advice_intensity",
  "provenance": "coach_groq_au",
  "pattern_type": "soa_letter_format",
  "target_text": "<|begin_of_text|>...<ADV response>...<|eot_id|>",
  "noise_text":  "<|begin_of_text|>...<BEN response>...<|eot_id|>",
  "user_prompt": "raw user question"
}
```
`target_text` = ADV (personal advice), `noise_text` = BEN (general information). Both are Llama 3.2-3B chat-formatted and ready for CAA extraction at Layer 14.

---

## Next Steps

1. **T2 replication (required for T2 promotion):** Independent session, different prompt pool, different random seed. Must be a different session from Entry 945 — T2 requires genuinely independent variance, not just a rerun.
2. **Characterise 2 CAA false negatives:** Two ADV pairs scored below threshold 0.1601 in Entry 945. Identify which `pattern_type` (conditional framing? SOA letter?) to determine if a second axis or revised corpus coverage is needed.
3. **Create `ensemble_config.json` for mean-pool concept dir:** `m-dad/concepts/asic_rg36_au_coach_v1_meanpool/` lacks an `ensemble_config.json`. Required before `m-dad detect` CLI can use this axis. Needs a custom mean-pool detection path in m-dad (standard prefill last-token projection will give incorrect scores).
4. **Benchmark comparison reference:** Entry 946 (T1) — GPT-OSS 120B on same 200-pair holdout: TPR=100% but FPR=14.0% (with detailed RG36 system prompt) / FPR=20.5% (no system prompt). CAA at FPR=0% outperforms prompted classifier on false positive rate.
