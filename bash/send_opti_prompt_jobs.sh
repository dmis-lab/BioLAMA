
#!/bin/bash
# WARNING: You MUST use bash to prevent errors


# These two table are common between the 2 scripts
MODEL=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "recobo/chemical-bert-uncased" "dmis-lab/biobert-base-cased-v1.2"  "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") # 
PROMPTS=("${DIR}/data/${TASK}/prompts/manual1.jsonl" "${DIR}/data/${TASK}/prompts/manual2.jsonl" "FS") # 


mkdir -p tmp_jobs

for i in "${!MODEL[@]}"
do

    for j in "${!PROMPTS[@]}"
    do
    sleep 1
    # Copy the template
    cp template_run_opti_prompt.sh tmp_jobs/run_opti_prompt_${i}_${j}.sh

    # replace parameters by values
    sed -i "s/REPLACEBYMODELINDEX/$i/g" tmp_jobs/run_opti_prompt_${i}_${j}.sh
    sed -i "s/REPLACEBYPROMPATH/$j/g" tmp_jobs/run_opti_prompt_${i}_${j}.sh
    sed -i "s/REPLACEBYJOBNAME/RunOptiPrompt$i$j/g" tmp_jobs/run_opti_prompt_${i}_${j}.sh

    # bash tmp_jobs/run_opti_prompt_${i}_${j}.sh

    done
done