TASK="np"
BATCHSIZE=16
DIR="."
PROMPT_TOKEN_LEN=20

MODELNAME=("PubMedBERT-full") # "ChemicalBERT" "BioBERT" "PubMedBERT" 
MODELTYPE=("BERT") # "BERT" "BERT" "BERT"
INITMETHODS=("independent") # "order" "confidence"
ITERMETHODS=("none") # "order" "confidence"
MODEL=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "recobo/chemical-bert-uncased" "dmis-lab/biobert-base-cased-v1.2" 

PROMPTS=("${DIR}/data/${TASK}/prompts/manual1.jsonl" "none") # "${DIR}/data/${TASK}/prompts/manual2.jsonl"

TRAIN_PATH=${DIR}/data/${TASK}/triples_processed/*/train.jsonl
DEV_PATH=${DIR}/data/${TASK}/triples_processed/*/dev.jsonl
TEST_PATH=${DIR}/data/${TASK}/triples_processed/*/test.jsonl

for i in "${!MODEL[@]}"
do
    echo "MODEL NAME = ${MODELNAME[i]}"
    echo "MODEL TYPE = ${MODELTYPE[i]}"
    echo "MODEL CODE = ${MODEL[i]}"

    for PROMPT in "${PROMPTS[@]}"
    do
        echo "PROMPT: ${PROMPT}"

        for INIT in "${INITMETHODS[@]}"
        do
        
        echo "TEST WITH INIT METHOD = ${INIT}"

        for ITER in "${ITERMETHODS[@]}"
        do
            echo "TEST WITH ITER METHOD = ${ITER}"
            
            if [ $PROMPT = "none" ]
            then
                echo "-- compute pronpt analysis"
                mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}

                python ${DIR}/BioLAMA/run_optiprompt.py \
                    --model_name_or_path ${MODEL} \
                    --train_path "${TRAIN_PATH}" \
                    --dev_path "${DEV_PATH}" \
                    --test_path "${TEST_PATH}" \
                    --num_mask 10 \
                    --init_method ${INIT} \
                    --iter_method ${ITER} \
                    --max_iter 10 \
                    --beam_size 5 \
                    --batch_size ${BATCHSIZE} \
                    --lr 3e-3 \
                    --epochs 10 \
                    --seed 0 \
                    --draft \
                    --pids rP703 \
                    --prompt_token_len ${PROMPT_TOKEN_LEN} \
                    --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}"

                echo "-- compute pronpt bias"
                mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}/MASKED
                
                python ${DIR}/BioLAMA/run_optiprompt.py \
                    --model_name_or_path ${MODEL} \
                    --train_path "${TRAIN_PATH}" \
                    --dev_path "${DEV_PATH}" \
                    --test_path "${DIR}/data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
                    --prompt_vector_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}_draft" \
                    --num_mask 10 \
                    --init_method ${INIT} \
                    --iter_method ${ITER} \
                    --max_iter 10 \
                    --beam_size 5 \
                    --batch_size ${BATCHSIZE} \
                    --seed 0 \
                    --draft \
                    --pids rP703 \
                    --prompt_token_len ${PROMPT_TOKEN_LEN} \
                    --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}/MASKED"
            else
                PROMPTFILE=$(basename $PROMPT)
                PROMPTNAME="${PROMPTFILE%.*}"
                echo "-- compute pronpt analysis"
                mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}

                python ${DIR}/BioLAMA/run_optiprompt.py \
                    --model_name_or_path ${MODEL} \
                    --train_path "${TRAIN_PATH}" \
                    --dev_path "${DEV_PATH}" \
                    --test_path "${TEST_PATH}" \
                    --num_mask 10 \
                    --init_method ${INIT} \
                    --iter_method ${ITER} \
                    --max_iter 10 \
                    --beam_size 5 \
                    --batch_size ${BATCHSIZE} \
                    --lr 3e-3 \
                    --epochs 10 \
                    --seed 0 \
                    --draft \
                    --pids rP703 \
                    --init_manual_template \
                    --prompt_path "${PROMPT}" \
                    --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}"

                echo "-- compute pronpt bias"
                mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/MASKED
                
                python ${DIR}/BioLAMA/run_optiprompt.py \
                    --model_name_or_path ${MODEL} \
                    --train_path "${TRAIN_PATH}" \
                    --dev_path "${DEV_PATH}" \
                    --test_path "${DIR}/data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
                    --prompt_vector_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}_imt_draft" \
                    --num_mask 10 \
                    --init_method ${INIT} \
                    --iter_method ${ITER} \
                    --max_iter 10 \
                    --beam_size 5 \
                    --batch_size ${BATCHSIZE} \
                    --seed 0 \
                    --draft \
                    --pids rP703 \
                    --init_manual_template \
                    --prompt_path "${PROMPT}" \
                    --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}/MASKED"
            fi
            done
        done
    done
done