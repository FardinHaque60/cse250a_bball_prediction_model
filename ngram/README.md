# ngram markov chain models trained using maximum likelihood estimation (MLE)

## ngram models
- `unigram.py` takes 2d array representing season data. ex: ```[
                [1, 0, 1, ...], # team record
                [0, 0, 1, ...], 
                ...
            ]```
    - `train_unigram()` trains the mle with this data to get initial and transition CPTs
    - `infer_season()` infers a season using sampling from CPTs to get sequence of 82 games from above generated CPTs

- `data_clustering.py` takes actual data and generates plot representing wins for all teams
    - used to determine clusters
    - depending on clusters, each one has its own mle model trained for it