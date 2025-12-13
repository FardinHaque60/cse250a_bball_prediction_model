# cse250a bball prediction model

predicting nba game outcomes using markov chain and hidden markov models on team-season data, using four factor (4f) statistics as observations.

for a full writeup of the project, see our report: [`Predicting Basketball Game Outcomes Using Markov Modeling`](Predicting%20Basketball%20Game%20Outcomes%20Using%20Markov%20Modeling.pdf)

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
- see `hmm/` to run the hmm (hidden markov model) via command line
