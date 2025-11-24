# cse250a bball prediction model

predicting basketball game outcomes using markov modeling

## running code

### create venv

- create `venv` using `python3 -m venv venv`
- activate using `source venv/bin/activate`
- install requirements using `pip install -r requirements.txt`

### data collection

- see `data/` to collect nba game data and/or combine partial nba game data into a single csv file

### running models

- see `notebooks/` to run the ngram and hmm models in notebook form
- see `ngram/` to run the ngram markov chain models via command line
- see `hmm/` to run the HMM (hidden markov model) via command lin
