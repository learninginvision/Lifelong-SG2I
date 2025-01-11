export GPU=0
CUDA_VISIBLE_DEVICES=$GPU python customconcept101/evaluate_IA.py \
    --sample_root logs/moe-10-task/reg\
    --target_path dataset-10task/ \
    --numgen 100 \
    --prompt_root bbox/dataset-10task-prompts \
    --concepts "dog,duck_toy,cat,backpack,teddybear,car,flower,lamp,shoes,bike" \
    --name "IA"