TASK="np"
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
MODELNAME=("ChemicalBERT")
MODELTYPE=("BERT" "BERT")
MODEL=("recobo/chemical-bert-uncased")
PROMPT_DIR_PATH=./data/${TASK}/prompts

for i in "${!MODEL[@]}"
do
    echo "MODEL NAME = ${MODELNAME[i]}"
    echo "MODEL TYPE = ${MODELTYPE[i]}"
    echo "MODEL CODE = ${MODEL[i]}"

    for PROMPT_PATH in "${PROMPT_DIR_PATH}"/*
    do
        echo "MANUAL PROMPT: ${PROMPT_PATH}"
        PROMPTFILE=$(basename $PROMPT_PATH)
        PROMPTNAME="${PROMPTFILE%.*}"

        echo "-- compute pronpt analysis"
        python ./BioLAMA/run_manual.py \
            --model_name_or_path ${MODEL[i]} \
            --prompt_path ${PROMPT_PATH} \
            --test_path "${TEST_PATH}" \
            --init_method independent \
            --iter_method none \
            --num_mask 3 \
            --max_iter 10 \
            --beam_size 15 \
            --batch_size 16 \
            --output_dir ./output2/${TASK}/${MODELNAME[i]}/${PROMPTNAME}

    done
done