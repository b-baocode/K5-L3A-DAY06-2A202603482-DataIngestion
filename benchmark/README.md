# Benchmark — openstax-ip-ch4-5

Frozen on 2026-11-13, **before any retrieval run**. Do not edit `questions.json` after looking at results; if the test set is changed, record it here.

- Corpus: Chapter 4 (Trademark) and Chapter 5 (Trade Secret) of OpenStax *Introduction to Intellectual Property* (CC BY 4.0). See [../data/manifest.json](../data/manifest.json).
- Questions: 20 in English (embedding model `BAAI/bge-small-en-v1.5` is English-only), 15 dev / 5 test.
- Types: direct 6, conditional 4, exception 3, paraphrase 2, filter 5 (3 dev, 2 test).
- Filter field: `ip_type` ∈ {`trademark`, `trade_secret`}, taken from the chapter title. A filter is an assumed user/context input given before retrieval. Only strategy C uses it.
- `gold_page` = physical page of the 50-page local PDF. q02 also accepts page 19 as gold.
- Verification: `python scripts/verify_benchmark.py` (from `starter/`) checks every quote on its gold page after `normalize_text()`; output in `starter/results/benchmark_verification.json`.

## Split

The test questions were chosen to cover topics that no dev question uses: employee mobility (q16), limitation period (q17), geographic marks (q18), trade-secret remedies (q19), trademark injunctions (q20).

Leakage risks:
- q20 (test) and the trademark remedies list on p22 are near the attorneys' fees distractor of q15 (dev).
- q19 (test) and q15 (dev) both concern trade-secret remedies (section 5.6), on different pages.
- Filter questions in dev and test use the same template idea (an ambiguous question disambiguated by `ip_type`).

## Selection criterion (fixed before running test)

1. Highest dev Evidence Hit@3.
2. If tied (difference ≤ 1 question): higher dev Hit@1.
3. Then manual grounding score, then fewer chunks / lower ingest time.

**Amendment (2026-11-13, after the dev run, before any test run):** dev left B and C tied (Hit@3 13/15, Hit@1 13/15). Manual grounding is labelled only on the test set, so it cannot be used to break a tie without looking at test. Rule 3 is therefore replaced by: *higher dev filter compliance (top-3 respects the question's filter), then resources.* Recorded in `starter/results/selection.json`.

**Test runs:** `--split test` was executed twice on 2026-11-13 with identical retrieval results. The second run only fixed how `artifacts/experiment_log.csv` reports embedding time; no strategy, chunking, model or question was changed after seeing test results.

## Authorship

Questions, answers and evidence were drafted with an AI assistant (Claude) from the extracted text, then checked automatically by the verification script. The student author should still read each item against the PDF; the grader is the author, which carries a risk of bias.
