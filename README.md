 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[CSN paper]: https://arxiv.org/abs/1909.09436

# Semantic Code Search

## Project Overview

[CodeSearchNet](https://arxiv.org/abs/1909.09436) is a collection of datasets and scripts
for the evaluation of semantic code search problem approaches. The authors also hosted a 
[benchmark](https://github.com/github/CodeSearchNet) 
with a public [leaderboard](https://wandb.ai/github/CodeSearchNet/benchmark)
to make the comparison of different approaches transparent. The models reported in the leaderboard were trained
to match method bodies with the corresponding documentation and evaluated on 99 natural language queries
labeled by human experts. Please, reference this [paper][CSN paper] for any further details regarding
the problem, the data collection or the benchmark.

## Setup

### Data

To reproduce our experiments you should download the archive with all the
resources (ASTs, graphs, embeddings):
https://drive.google.com/file/d/1XPYViMkZTYae0DkbzTg3RB6-mG-x043E.

Alternatively, you can download the CodeSearchNet benchmark data by running:

`$ script/setup`

and doing preprocessing yourself following the instructions below.

### Environment

```
$ conda create -n codesearch python=3.6 tensorflow-gpu=1.15.0
$ conda activate codesearch
$ pip install -r requirements.txt
```

### Training

We train our models on python data. 

To train the NBoW model, run:

```
$ python src/train.py --model neuralbow trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test
```

For the training commands of the tree models, see `./run_trees.sh`.

To train the graph models, see `./run_graphs.sh` and `./run_graphs_plain.sh`.

### Evaluation

After training is complete, you will get a checkpoint in the specified directory
(e.g., `src/trained_models/$_model_best.pkl.gz`).

To rerun the MRR testing, launch from `src` dir the following command:

```
$ python test.py trained_models/$_model_best.pkl.gz
```

Run the following commands from root for prediction and evaluation on the 99 natural language queries:

```
$ python src/predict.py -m src/trained_models/$_model_best.pkl.gz -p predictions.csv
$ python src/relevanceeval.py resources/annotationStore.csv predictions.csv
```

### AST parsing

For AST parsing we use [tree-sitter](https://github.com/tree-sitter/tree-sitter).
To work with parsers, do the following:

1. Clone the appropriate tree-sitter parsers to `src/procesing/tree-sitter-languages`
2. Add information about the cloned parsers to [setup_tree_sitter.py](src/processing/setup_tree_sitter.py)
3. Run [setup_tree_sitter.py](src/processing/setup_tree_sitter.py)
4. Add a wrapper to [tree_sitter_parsers.py](src/processing/ast_parsers/tree_sitter_parsers.py)
5. Add the language to [run.py](src/processing/run.py)

### Graph parsing

For graph parsing we use [typilus fork](https://github.com/JetBrains-Research/typilus).
To work with graph parsers, do the following:

1. Clone the aforementioned repository to `src/processing/typilus`

## Models

Currently, ten models are adapted or re-implemented. All these models are widely used in the SE domain for different
for different problems (e.g. variable naming, variable misuse prediction, method name prediction).
* `NBOW`: Neural Bag-of-Words Model ([Sheikh et al.](https://www.aclweb.org/anthology/W16-1626))

TODO: finish the list

### Pretrained models

You can load for the testing/evaluation using the run ids from the [Weights & Biases](https://wandb.ai).
The pretrained models in the same order as they appear in the paper are:

`NBOW`: [msazanovich/CodeSearchNet-src/33jlj8pn](https://wandb.ai/msazanovich/CodeSearchNet-src/runs/33jlj8pn)

`NBOW + AST`: [msazanovich/CodeSearchNet-src/37f1j1l2](https://wandb.ai/msazanovich/CodeSearchNet-src/runs/37f1j1l2)

`AST node2vec`: [msazanovich/CodeSearchNet-src/ihap35bh](https://wandb.ai/msazanovich/CodeSearchNet-src/runs/ihap35bh)

`TBCNN`: [msazanovich/CodeSearchNet-src/77y9fj1y](https://wandb.ai/msazanovich/CodeSearchNet-src/runs/77y9fj1y)

`ASTNN`: [egor-bogomolov/semantic-code-search-src/3qzilyhq](https://https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/3qzilyhq) 

`NBOW + graph`: [egor-bogomolov/semantic-code-search-src/3nq3g667](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/3nq3g667)

`Graph node2vec`: [msazanovich/CodeSearchNet-src/1r3y07gk](https://wandb.ai/msazanovich/CodeSearchNet-src/runs/1r3y07gk)

`Transformer + types`: [egor-bogomolov/semantic-code-search-src/2nf2sb3b](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/2nf2sb3b)

`GREAT`: [egor-bogomolov/semantic-code-search-src/334uyl5k](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/334uyl5k)

`GGNN`: [egor-bogomolov/semantic-code-search-src/2c1vwty5](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/2c1vwty5)

`GRU-GGNN`: [egor-bogomolov/semantic-code-search-src/1atdvf5p](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/1atdvf5p)

`Transformer-GGNN`: [egor-bogomolov/semantic-code-search-src/f35yfbis](https://wandb.ai/egor-bogomolov/semantic-code-search-src/runs/f35yfbis)

## Experimental results


| Model               | Test MRR  | NDCG within | NDCG full  |
|---------------------|-----------|-------------|------------|
| NBOW                | 64.1      | 61.6        | 47.1       |
| BiGRU               |           |             |            |
| Transformer         |           |             |            |
| NBOW + AST          | 65.1      | 63.0        | 49.2       |
| AST node2vec        | 64.4      | 62.4        | 49.3       |
| TBCNN               | 63.7      | 60.6        | 43.8       |
| ASTNN               | 65.5      | 55.4        | 44.6       |
| NBOW + graph        | 66.4      | 53.9        | 40.3       |
| Graph node2vec      | 64.8      | 62.8        | 45.1       |
| Transformer + types | 66.3      | 54.2        | 40.1       |
| GREAT               | 66.6      | 54.9        | 40.1       |
| GGNN                | 66.5      | 55.3        | 41.1       |
| GRU-GGNN            | 66.0      | 53.6        | 39.3       |
| Transformer-GGNN    | 66.6      | 54.4        | 39.9       |
