# maximum likelihood estimation (MLE) model

## mle
- `mle_model.py` takes 2d array representing season data. ex: ```[
                ["W", "L", "W", ...], # team record
                ["L", "L", "W", ...], 
                ...
            ]```
    - `train_mle_model()` trains the mle with this data to get initial and transition CPTs
    - `infer_season()` infers a season using sampling from CPTs to get sequence of 82 games from above generated CPTs

- `mle_cluster.py` takes actual data and generates plot representing wins for all teams
    - used to determine clusters
    - depending on clusters, each one has its own mle model trained for it