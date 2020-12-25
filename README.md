# Semantic Code Search

## Setup

### Data

To reproduce our experiments you should download the archive with all the
resources (ASTs, graphs, embeddings:
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
