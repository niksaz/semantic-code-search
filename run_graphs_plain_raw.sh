python src/train.py --model graphnbow-raw trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model graphnbow trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer10-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer10-plain-raw trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn-plain trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn-plain-raw trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

