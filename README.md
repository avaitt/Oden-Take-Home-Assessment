# Finding precursors to rare events: Azure Predictive Maintenance

Take-home for Oden Technologies (Forward Deployed Data Scientist), **Option 1**. It is a small version of the
workflow in Oden's post, [Agentic Root Cause Analysis](https://engineering.oden.io/blog/agentic-root-cause-analysis):
brainstorm candidate precursors, turn each into a simple fixed rule, test them honestly, and keep up to three.

## Read the notebooks, in order

| Notebook | Brief section | What it covers |
|---|---|---|
| `notebooks/01_dataset_and_parameters.ipynb` | Dataset + Task 1 | the five raw CSVs; the dataset profile (events, sampling, gaps, regimes, counters); a timestamp audit; where the warning signs live; setting H, W and the event; confidence |
| `notebooks/02_candidates.ipynb` | Task 2 | 10 candidate precursors, each with a physical story and a yes/no detector |
| `notebooks/03_ml_benchmark.ipynb` | Task 2 ("an ML model") | the ML models used as an idea source and a ceiling, and why the simple rules win |
| `notebooks/04_test_and_select.ipynb` | Task 3 | lift, precision and coverage for every candidate; guardrails; the shortlist and why it wins; **one F1 grid over 100 (H, W) windows** |
| `notebooks/05_audit.ipynb` | "What we look for" | trying to break the results (lucky picks, leakage, counters, time drift, regimes, clustering) |
| `notebooks/06_writeup.ipynb` | Task 4 | **start here**: a standalone summary with the answer, key terms (incl. the gates), where every number comes from (the files notebooks 01–05 save), each notebook's key findings, then the shortlist, rejected candidates, assumptions, data source, design choices and next steps |

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter and **run them in order, 01 → 06** (Run All in each). Notebooks 01–05
each compute their analysis live and, in their last cell, save their key results to `outputs/`; notebook 06,
the write-up, reads those five files and computes nothing.

| Run | Time | Saves |
|---|---|---|
| 01 Dataset and parameters | seconds (plus a one-off ~31 MB download) | `outputs/01_dataset_and_parameters.json` |
| 02 Candidates | seconds | `outputs/02_candidates.json` |
| 03 ML benchmark | ~2 min | `outputs/03_ml_benchmark.json` |
| 04 Test and select | ~10 min | `outputs/04_test_and_select.json` |
| 05 Audit (reads 04's file) | ~1 min | `outputs/05_audit.json` |
| 06 Write-up (reads all five) | seconds | — |

To try other settings, change `Config(...)` (H, W, event) at the top of each notebook and re-run them all; the
write-up warns if the notebooks were run with different settings or if a file is out of date.

**Data:** [Microsoft Azure Predictive Maintenance on Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
(version 3, five CSVs). It downloads on first use with
`kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")`; no account is needed.

**Code:** three helper files in `src/`, read in this order: `data.py` (settings, loading, windows) →
`detectors.py` (rules and candidates) → `evaluation.py` (metrics, selection, audits, ML benchmark, saving results).

## Result

**Goodness = F1** of precision and coverage. The window is **"will it fail within the next 48 hours?"**
(H = 0, W = 48), chosen by F1 on a grid of 100 windows (notebook 04, section 6). **My recommendation for
deployment is H = 12, W = 36**, so every alert leaves half a day to act. That costs ~0.01 F1 and changes no
detector (see "My decisions" below). Every choice was made on 50 dev machines; the numbers
below are from the other **50 holdout machines** ("any failure", 350 events), with 95% intervals.

| # | Precursor | Rule | F1 | Precision | Coverage | Lead |
|---|---|---|---|---|---|---|
| 1 | **C09 Matched signature** | a part's **own** signal (6 h mean) beyond its machine's P99 (P1 for rotation) **and** that part's **own** error code logged in the last 24 h | **0.94** | 92% [89, 94] | 96% [94, 98] | ~23 h |
| 2 | C08 Drift + any error | any signal's 6 h mean beyond its machine's median ± 3.72 MAD-σ **and** any error code in the last 48 h | 0.76 | 62% [58, 67] | 98% [97, 99] | ~24 h |
| 3 | C10 Drift on an aging part | a part's own signal drifting **and** the part older than its median time since replacement | 0.56 | 41% [37, 45] | 91% [87, 94] | **~43 h** |

A random alarm would be right 4% of the time. Each part's signal drifts ~48 h before it fails, but ~470
look-alike drifts per part never end in a failure. **The part's own error code tells them apart.** That is
why C09 wins. Nothing is detectable more than ~1.5 days ahead. The data is synthetic and unusually clean, so
treat absolute numbers as optimistic.

## My decisions

I directed the analysis and made its key calls; the AI assistant wrote most of the code to my specification.

* **F1 as the measure of goodness.** Lift, precision and coverage can each be gamed on their own: a detector
  that almost never fires gets a huge lift, and one that is always on gets full coverage. I chose F1 of
  precision and coverage as the single score because it is high only when most failures are caught *and* most
  alarms are real, so it is much harder to game.
* **Using F1 to choose H and W.** Rather than picking a window by convention, I had H and W chosen by the same
  F1 score, on a proper grid of 100 windows (10 leads × 10 lengths), scored on dev machines only.
* **W = 48 from the data.** Looking at the data myself (notebook 01, section 6), I saw that each part's own
  signal starts drifting about 48 h before it fails and stays shifted until the failure. So the window has to
  reach back 48 h to see the whole precursor, rather than the common one-day window.
* **H = 12, W = 36 as my recommendation, even though H = 0, W = 48 has the higher F1.** With H = 0, an alert
  one hour before a failure counts as a success, which leaves no time to act. I decided the window needs a
  buffer. H = 12 is reasonable because:
  - it guarantees half a day of warning, about one maintenance shift: enough to schedule a crew and a part;
  - W = 36 still reaches back to where the drift starts (48 h before the failure), and still contains the
    part's own error code (logged 24 h before), so the window covers the whole precursor;
  - the best detector alerts ~23 h ahead and the early-warning one ~43 h ahead, so a 12 h buffer still gives
    them credit;
  - it costs only ~0.01 F1 (0.936 vs 0.945), less than the noise (~±0.02), and changes no detector or alert;
  - a larger buffer would not work: F1 collapses once the window starts 36 h or more ahead.
* **The ML benchmark as a guide and a ceiling.** I had the ML models kept and documented as their own notebook
  (03), used to guide which signals to build rules on and to check that the simple rules are not leaving much
  on the table.
* **Simplicity and clarity.** I pushed for "less is more": one notebook per task in the brief, ten candidates
  instead of nineteen, one F1 metric instead of many, redundant charts and analyses removed, and a plain-text
  conclusion at the end of every section.
* **Audits I asked for:** an hour-by-hour check of how failures are timestamped (which showed that a
  24 h lead isn't needed to avoid leakage), and repeated checks of the work against every requirement in the brief.

## Where the AI assistant had to be corrected

Built with an AI coding assistant (Claude). Each of these was caught by a check, not assumed:

1. **Fake lift:** smoothing both rates gave a detector that never fires a lift of 47. Only the normal rate is
   smoothed now.
2. **Count thresholds:** "error count > P90" silently meant "≥ 2 errors", which only comp2 can meet. P80
   ("≥ 1 error") was added.
3. **Contaminated "normal":** ~470 look-alike drifts per part push P99 into the anomaly cluster. This led to
   robust thresholds, and to the main finding.
4. **Redundancy rule:** the first version discarded earlier warnings for the same failures. It now drops only
   detectors that are the same alarm.
5. **Null p-value floor:** an empirical p can't go below 1/(n+1). The gate now uses the null z-score there.
6. **Counter leak:** "hours since maintenance" passed every statistical gate, then collapsed with past-only
   thresholds. This led to the causal gate.
7. **A claim the evidence contradicted:** the first write-up said a grid confirmed W = 24. It didn't.
8. **Code pairing:** a fixed enrichment cut-off paired error5 with the wrong parts. A relative rule fixed it.
9. **H = 24 was never tested:** an hour-by-hour audit removed its rationale. With the user's guidance (choose
   by F1), the F1 grid chose H = 0, W = 48.
10. **Misleading rule text:** rotation's rule printed "> P99" instead of "< P1".
11. **Engineering slips:** a pandas name clash, read-only arrays, a masked leakage demo, and a freshness check
    fooled by file timestamps.

**For the submission:** export this AI conversation and attach it next to this README.

## If there were more time

* Ask whether the look-alike drifts precede maintenance that averted a failure. If so, some "false alarms"
  are really saves.
* Train a per-part ML model (one output per component) and score it with the brief's alert-level metrics.
* Replace F1 with an expected cost once the costs of a false alarm and of a missed failure are known.
* Re-check error-code timing with a real, non-batched log, and try a minute-level dataset.
