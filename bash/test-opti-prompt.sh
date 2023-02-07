TASK="np"
MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
PROMPT_PATH=./data/${TASK}/prompts/manual1.jsonl
TRAIN_PATH=./data/${TASK}/triples_processed/rP703/train.jsonl
DEV_PATH=./data/${TASK}/triples_processed/rP703/dev.jsonl
TEST_PATH=./data/${TASK}/triples_processed/rP703/test.jsonl

python ./BioLAMA/run_optiprompt.py \
    --model_name_or_path ${MODEL} \
    --train_path "${TRAIN_PATH}" \
    --dev_path "${DEV_PATH}" \
    --test_path "${TEST_PATH}" \
    --prompt_path ${PROMPT_PATH} \
    --num_mask 10 \
    --init_method confidence \
    --iter_method none \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --lr 3e-3 \
    --epochs 10 \
    --seed 0 \
    --pids rP703 \
    --prompt_token_len 20 \
    --init_manual_template \
    --output_dir ./output3/${TASK}/optiprompt/PubMedBERT/manual1