# code-search-net

## Setup

### Data

`$ script/setup`

### Environment

```
$ conda create -n codesearch python=3.6 tensorflow-gpu=1.12.0
$ conda activate codesearch
$ pip install -r requirements.txt
```

### Training

To train only on the python data, run:

```
$ python src/train.py --model neuralbow trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test
```

### AST parsing

For AST parsing we use [tree-sitter](https://github.com/tree-sitter/tree-sitter).
To work with parsers, do the following:

1. Clone the appropriate tree-sitter parser in `src/parsing/tree-sitter-languages`
2. Add information about it to [setup_tree_sitter.py](src/processing/setup_tree_sitter.py)
3. Add the language to [utils.py](src/processing/utils.py)
4. Create a wrapper (see [PythonAstParser](src/processing/ast_parsers/python_ast_parser.py) for example)  
