set -x

DRYRUN=''

python src/train.py $DRYRUN --model nbowtypesast trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test

python src/train.py $DRYRUN --model node2vecast trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test

python src/train.py $DRYRUN --model tbcnnast \
    --hypers-override '{"batch_size": 100}' trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test

python src/train.py $DRYRUN --model node2vecgraphs trained_models \
    resources/data/python/final/jsonl/train \
    resources/data/python/final/jsonl/valid \
    resources/data/python/final/jsonl/test
