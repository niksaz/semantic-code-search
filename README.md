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

## Training

To train only on the python data, run:

```
$ python src/train.py --model neuralbow trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test
```
