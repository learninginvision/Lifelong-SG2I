
export GPU=3
strArray=("dog" "duck_toy" "cat" "backpack" "teddybear" "car" "flower" "lamp" "shoes" "bike")
task_strArray=("photo of a <new1> dog" "photo of a <new2> duck_toy" "photo of a <new3> cat" "photo of a <new4> backpack" "photo of a <new5> teddybear" "photo of a <new6> car" "photo of a <new7> flower" "photo of a <new8> lamp" "photo of a <new9> shoes" "photo of a <new10> bike")
promt_strArray=("<new1> dog" "<new2> duck_toy" "<new3> cat" "<new4> backpack" "<new5> teddybear" "<new6> car" "<new7> flower" "<new8> lamp" "<new9> shoes" "<new10> bike")

index=0

for value in "${strArray[@]}"
do 
    index=$((index + 1))
    echo "Index $index: $value"
    echo "Index $index: ${task_strArray[index-1]}"
    
    CUDA_VISIBLE_DEVICES=$GPU  python -u sample_lora_continual_bbox.py  \
        --prompt "${promt_strArray[$index-1]}" \
        --n_samples 1 \
        --n_iter 25 \
        --ddim_steps 50 \
        --delta_ckpt  "logs/moe-10-task-self-ablations/checkpoints/delta_epoch=last.ckpt" \
        --ckpt /data_8T/cys/continual_lora_moe/sd-v1-5-emaonly.ckpt \
        --concept "$value"\
        --task_id 200\
        --task_prompt "${task_strArray[$index-1]}" \
        --skip_grid

done
