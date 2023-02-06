TASK="np"
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
MODELNAME=("ChemicalBERT" "BioBERT" "PubMedBERT")
MODELTYPE=("BERT" "BERT" "BERT")
MODEL=("recobo/chemical-bert-uncased" "dmis-lab/biobert-base-cased-v1.2" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
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
        echo "./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}"

        echo "-- compute pronpt bias"
        echo "./data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl"
        echo "./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/MASKED"

    done
done
