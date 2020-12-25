python src/train.py --model ggnn trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn-ggnn-sandwich trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer-ggnn-sandwich trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model great trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model great10 trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model transformer10 trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test

python src/train.py --model rnn trained_models \
     resources/data/python/final/jsonl/train \
     resources/data/python/final/jsonl/valid \
     resources/data/python/final/jsonl/test