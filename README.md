# Finding precursors to rare events: Azure Predictive Maintenance

Take-home for Oden Technologies (Forward Deployed Data Scientist), **Option 1**. It is a small version of the
workflow in Oden's post, [Agentic Root Cause Analysis](https://engineering.oden.io/blog/agentic-root-cause-analysis):
brainstorm candidate precursors, turn each into a simple fixed rule, test them honestly, and keep up to three.

## Website

**[avaitt.github.io/Oden-Take-Home-Assessment](https://avaitt.github.io/Oden-Take-Home-Assessment/)**: the findings
in a browser, no setup needed.

* **Write-up:** notebook 06 as a web page: the answer, the three rules written out, and how they were built and tested.
* **Rule explorer:** pick any window (H, W) and event (any failure or a single part) and see how C09, C08 and C10
  score on the holdout machines: F1, precision, coverage, lift, alerts per machine-month and lead time.
* **Machine explorer:** pick any machine and sensor to see its year of readings, with every error code, part
  replacement and failure marked.

## Read the notebooks, in order

| Notebook | Brief section | What it covers |
|---|---|---|
| `notebooks/01_dataset_and_parameters.ipynb` | Dataset + Task 1 | the five raw CSVs; the dataset profile (events, sampling, gaps, regimes, counters); a timestamp audit; where the warning signs live; setting H, W and the event; confidence |
| `notebooks/02_candidates.ipynb` | Task 2 | 10 candidate precursors, each with a physical story and a yes/no detector |
| `notebooks/03_ml_benchmark.ipynb` | Task 2 ("an ML model") | the ML models used as an idea source and a ceiling, and why the simple rules win |
| `notebooks/04_test_and_select.ipynb` | Task 3 | lift, precision and coverage for every candidate; guardrails; the shortlist and why it wins; **one F1 grid over 100 (H, W) windows**, which supports the window used throughout, H = 12, W = 36 |
| `notebooks/05_audit.ipynb` | "What we look for" | trying to break the results (lucky picks, leakage, counters, time drift, regimes, clustering) |
| `notebooks/06_writeup.ipynb` | Task 4 | **start here**: a standalone summary with the answer, key terms (incl. the gates), each notebook's key findings, the three rules written out, rejected candidates, how the rules were tested (no leakage), assumptions, design choices and next steps |

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
| 04 Test and select | ~9 min | `outputs/04_test_and_select.json` |
| 05 Audit (reads 04's file) | seconds | `outputs/05_audit.json` |
| 06 Write-up (reads all five) | seconds | — |

To try other settings, change `Config(...)` (H, W, event) at the top of each notebook and re-run them all; the
write-up warns if the notebooks were run with different settings or if a file is out of date.

**Website build:** the website (above) is built into `docs/` by `python site/build_site.py` (~2 min, after running
the notebooks); GitHub Pages serves it from there.

**Data:** [Microsoft Azure Predictive Maintenance on Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
(version 3, five CSVs). It downloads on first use with
`kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")`; no account is needed.

**Code:** three helper files in `src/`, read in this order: `data.py` (settings, loading, windows) →
`detectors.py` (rules and candidates) → `evaluation.py` (metrics, selection, audits, ML benchmark, saving results).

## Result

**Goodness = F1** of precision and coverage. Alerts are scored on the window **H = 12, W = 36**: an alert counts
if it comes between 48 and 12 hours before a failure, so every alert that counts leaves at least half a day to
act. The rules were **built and tuned on 50 dev machines and tested on the other 50 holdout machines**, which
played no part in building them (no leakage). Numbers below are on the holdout machines ("any failure",
350 events), with 95% intervals for F1.

| # | Precursor | Rule (alert when both hold) | F1 | Precision | Coverage | Lead |
|---|---|---|---|---|---|---|
| 1 | **C09 Matched signature** | a part's **own** sensor (6 h average) more than 3.09 spreads beyond its machine's usual level **and** that part's **own** error code logged in the last 24 h | **0.94** [0.92, 0.96] | 90% | 99% | ~24 h |
| 2 | C08 Drift + any error | any sensor (6 h average) more than 3.72 spreads beyond its machine's usual level **and** any error code in the last 48 h | 0.74 [0.71, 0.78] | 63% | 91% | ~24 h |
| 3 | C10 Drift on an aging part | a part's own sensor more than 3.72 spreads beyond its usual level **and** the part older than usual since its last replacement (31–34 days) | 0.56 [0.52, 0.60] | 41% | 89% | **~43 h** |

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
* **A window that reaches back 48 h, from the data.** Looking at the data myself (notebook 01, section 6), I saw
  that each part's own signal starts drifting about 48 h before it fails and stays shifted until the failure. So
  the window has to reach back 48 h to see the whole precursor, rather than the common one-day window.
* **H = 12, W = 36, even though a window with no buffer scored slightly higher.** With H = 0, an alert one hour
  before a failure counts as a success, which leaves no time to act. I decided the window needs a buffer.
  H = 12 is reasonable because:
  - it guarantees half a day of warning, about one maintenance shift: enough to schedule a crew and a part;
  - W = 36 still reaches back to where the drift starts (48 h before the failure), and still contains the
    part's own error code (logged 24 h before), so the window covers the whole precursor;
  - on the F1 grid it scores 0.936 against 0.948 for the best window, within the noise (~±0.02);
  - a larger buffer would not work: F1 collapses once the window starts 36 h or more ahead.
* **The ML benchmark as a guide and a ceiling.** I had the ML models kept and documented as their own notebook
  (03), used to guide which signals to build rules on and to check that the simple rules are not leaving much
  on the table.
* **Simplicity and clarity.** I pushed for "less is more": one notebook per task in the brief, ten candidates
  instead of nineteen, one F1 metric instead of many, redundant charts and analyses removed, and a plain-text
  conclusion at the end of every section.
* **No leakage.** I questioned how the holdout machines were scored and found that their per-machine thresholds
  were built using their failure dates. Now the rules are built and tuned on dev machines only, a holdout
  machine's baseline never uses its failure dates, and the ML feature importances that fed the brainstorm are
  measured on dev machines.
* **The website.** I had the write-up published as a site with a rule explorer (score the rules on any window
  and event) and a machine explorer (any machine's year of any sensor).
* **Audits I asked for:** an hour-by-hour check of how failures are timestamped (which showed that a
  24 h lead isn't needed to avoid leakage), and repeated checks of the work against every requirement in the brief.

## Where the AI assistant had to be guided

Built with an AI coding assistant (Claude). Each of these was caught and corrected by me:

1. Claude scored the H × W grid on several metrics. I had it scored by F1 alone, for simplicity and coherence.
2. Claude set a 24-hour lead on an untested assumption. I questioned it; the timestamp audit showed it had no basis.
3. Claude chose H = 0, which leaves no time to act on an alert. I required a buffer, which led to H = 12, W = 36.
4. Claude built holdout machines' thresholds from their own failure dates, and measured the ML feature importances on holdout machines. I caught the leakage, and both now use dev machines only.
5. Claude claimed a 12-hour buffer "costs ~0.01 F1", having checked it only on the dev grid. I tested it on the rule explorer and found the holdout picture was different.
6. Claude measured some comparisons of C08, C09 and C10 on a different window from the chosen one. I had 12/36 used everywhere.
7. Claude put the re-tuning step in the write-up and the website. I moved it to notebook 04 (test and select), where it belongs.
8. Claude computed the H × W grid in two notebooks. I removed the duplicate.
9. Claude's explanations leaned on jargon ("roll6", "robust", "alert lines") and confusing tables. I had them rewritten in plain language.

## If there were more time

* Ask whether the look-alike drifts precede maintenance that averted a failure. If so, some "false alarms"
  are really saves.
* Train a per-part ML model (one output per component) and score it with the brief's alert-level metrics.
* Replace F1 with an expected cost once the costs of a false alarm and of a missed failure are known.
* Re-check error-code timing with a real, non-batched log, and try a minute-level dataset.
* Compute each machine's usual level from past readings only (a rolling baseline), as a live system would.
