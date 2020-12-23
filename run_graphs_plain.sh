python src/train.py --model ggnn-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn-ggnn-sandwich-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer-ggnn-sandwich-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model great-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model great10-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer10-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model graphnbow trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test