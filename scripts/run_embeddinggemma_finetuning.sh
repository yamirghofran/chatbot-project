
export RAW="https://pub-eecdafb53cc84b659949b513e40369d2.r2.dev/files/md5/95/56aaf6c17f076302a275361d4b73ce"
export TXT="https://pub-eecdafb53cc84b659949b513e40369d2.r2.dev/files/md5/b7/b4993f39c4df13ce52cf1f3aa79e4a"
export OUT="finetuning_results/embeddinggemma_unsloth_run1"
uv run python scripts/finetune_embeddinggemma.py \
--raw-books-source "$RAW" \
--book-texts-source "$TXT" \
--output-dir "$OUT" \
--batch-size 64 \
--max-pairs 100 \
--gradient-accumulation-steps 4 \
--max-seq-length 512 \
--eval-max-queries 1000 \
--save-merged-16bit \
--batch-size 64 \
--max-pairs 20000 \ 
--no-eval-baseline \
--eval-max-queries 1000 \
--eval-query-batch-size 32 \
--eval-similarity-matrix-mb 64 \
--eval-steps 1000000
