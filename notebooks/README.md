# notebooks for ngram and hmm models

- **run_ngram_models.ipynb**
  - demonstrates training and inference for unigram, bigram, and trigram markov chain models
  - uses helper functions from the `ngram` package (for example `train_unigram`, `train_bigram`, `train_trigram`)
  - loads season win/loss sequences derived from `data/allseasons.csv`
  - compares model predictions to actual seasons and reports sequence-level accuracy

- **run_hmm_models.ipynb**
  - demonstrates supervised training and evaluation of discrete-emission hmm models for nba game outcomes
  - uses preprocessing utilities from `hmm/preprocess.py` to:
    - read games from `data/allseasons.csv`
    - standardize four-factor stats and bin them into discrete observation symbols
    - build team-season sequences and split into train / test sets
  - trains an hmm via maximum likelihood using known win/loss states
  - decodes test sequences with viterbi and evaluates performance
  - includes visualizations:
    - histogram of per-sequence accuracies
    - confusion matrix heatmap
    - example season plot of true vs predicted states
    - per-season accuracy aggregation and bar chart
