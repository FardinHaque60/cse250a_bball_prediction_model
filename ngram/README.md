# ngram markov chain models trained using maximum likelihood estimation (MLE)

## ngram models
- `{uni,bi,tri}gram.py` library files containing train and inference functions
    - `train_{uni,bi,tri}gram()` trains ngram model with mle to get initial and transition CPTs. takes 2d array representing season data. ex: ```[
                [1, 0, 1, ...], # team record
                [0, 0, 1, ...], 
                ...
            ]```
    - `infer_{uni,bi,tri}gram_season()` infers a season using sampling from CPTs to get sequence of 82 games from above generated CPTs

- `data_clustering.py` takes actual data and generates plot representing wins for all teams
    - used to determine clusters
    - depending on clusters, each one has its own mle model trained for it

- `ngram_runner.py` modify and run this file to test a ngram library file