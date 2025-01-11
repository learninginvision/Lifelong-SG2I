ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done


python -u  train_lora_continual.py \
        --base configs/custom-diffusion/finetune_addtoken_loramoe_continual.yaml  \
        -t --gpus 0, \
        --resume-from-checkpoint-custom sd-v1-5-emaonly.ckpt \
        --datapath dataset-10task\
        --reg_datapath gen_reg101 \
        --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>+<new6>+<new7>+<new8>+<new9>+<new10>" \
        --name "moe-cys"\
        --batch_size 2\
        --num_tasks 10\
        --concepts "dog,duck_toy,cat,backpack,teddybear,car,flower,lamp,shoes,bike" \
        --device "cuda:0"