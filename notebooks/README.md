# notebooks for ngram, hmm, and threshold models

- **dataset.ipynb**
  - summarizes the nba regular-season dataset used across all experiments
  - describes the csv schema for `data/allseasons.csv` and basic dataset statistics (number of seasons, teams, and games)
  - illustrates how we compute per-season z-score normalization of the four factors (efg%, tov%, orb%, ft/fga)
  - reports the optimized four-factor weights learned via logistic regression and compares them to dean oliver's baseline weights

- **run_ngram_models.ipynb**
  - example workflow for training and evaluating unigram, bigram, and trigram markov (ngram) models on win/loss sequences
  - uses helpers from the `ngram` package to load tiered team-season sequences from `data/allseasons.csv`
  - configures a specific clustering scheme via constants such as `CLUSTERS`, `TIER`, and win-range thresholds
  - trains ngram models on historical seasons and evaluates sequence-level accuracy on held-out seasons

- **run_hmm_models.ipynb**
  - example workflow for training and evaluating discrete-emission hmm models for nba game outcomes
  - uses `hmm/preprocess.py` to build team-season sequences:
    - reads `data/allseasons.csv`
    - standardizes four-factor stats, applies either dean oliver or optimized weights, and bins into discrete observation symbols
    - splits sequences into train and test sets using a holdout-season scheme (e.g., 2018 and 2024 as test seasons)
  - trains both supervised and unsupervised hmms using utilities from `hmm/hmm_model.py`
  - decodes test sequences with viterbi and reports per-game accuracy, along with plots such as accuracy histograms and confusion matrices

- **run_hmm_grid_search.ipynb**
  - performs a grid search over key hmm hyperparameters
  - sweeps over numbers of hidden states and observation bins, optionally mixing supervised and unsupervised training depending on the state count
  - caches preprocessed sequences for different bin counts and records evaluation metrics for each configuration

- **run_threshold_model.ipynb**
  - example workflow for threshold-based baselines built on four-factor scores
  - uses utilities from the `threshold` package to construct tiered datasets from `data/allseasons.csv`
  - trains simple threshold rules per tier and evaluates their predictive performance on held-out seasons
