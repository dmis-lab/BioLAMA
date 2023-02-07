TASK="np"
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
MODELNAME=("ChemicalBERT" "BioBERT" "PubMedBERT" "PubMedBERT-full")
MODELTYPE=("BERT" "BERT" "BERT" "BERT")
INITMETHODS=("independent" "order" "confidence")
ITERMETHODS=("none" "order" "confidence")
MODEL=("recobo/chemical-bert-uncased" "dmis-lab/biobert-base-cased-v1.2" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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

        for INIT in "${INITMETHODS[@]}"
        do
        
        echo "TEST WITH INIT METHOD = ${INIT}"

        for ITER in "${ITERMETHODS[@]}"
        do
            echo "TEST WITH ITER METHOD = ${ITER}"
            
            echo "-- compute pronpt analysis"
            mkdir -p ./output/${TASK}/manual/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}

            python ./BioLAMA/run_manual.py \
                --model_name_or_path ${MODEL[i]} \
                --prompt_path ${PROMPT_PATH} \
                --test_path "${TEST_PATH}" \
                --init_method ${INIT} \
                --iter_method ${ITER} \
                --num_mask 10 \
                --max_iter 10 \
                --beam_size 5 \
                --batch_size 16 \
                --output_dir ./output/${TASK}/manual/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER} > ./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/log.log

            echo "-- compute pronpt bias"
            mkdir -p ./output/${TASK}/manual/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/MASKED

            python ./BioLAMA/run_manual.py \
                --model_name_or_path ${MODEL[i]} \
                --prompt_path ${PROMPT_PATH} \
                --test_path "./data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
                --init_method ${INIT} \
                --iter_method ${ITER} \
                --num_mask 10 \
                --max_iter 10 \
                --beam_size 5 \
                --batch_size 16 \
                --output_dir ./output/${TASK}/manual/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/MASKED > ./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/MASKED/log.log
            done
        done
    done
done