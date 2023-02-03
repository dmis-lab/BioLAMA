TASK="np"
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
MODEL="dmis-lab/biobert-base-cased-v1.2"
PROMPT_PATH=./data/${TASK}/prompts/manual1.jsonl

python ./BioLAMA/run_manual.py \
    --model_name_or_path ${MODEL} \
    --prompt_path ${PROMPT_PATH} \
    --test_path "${TEST_PATH}" \
    --init_method confidence \
    --iter_method none \
    --num_mask 10 \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --output_dir ./output/${TASK}_manual