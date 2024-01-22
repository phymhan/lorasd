export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./images_db/dog"
# export OUTPUT_DIR="./outputs/debug"
# export CLASS_DATA_DIR="./images_class/dog/samples"
PROMPTS="photo of a sks dog<SEP>photo of a sks dog in Times Square<SEP>a sks dog swimming pool<SEP>a sks dog made of Lego blocks<SEP>a sks dog in Minecraft style<SEP>watercolor painting of a sks dog"

GPU_ID=1

LR="1e-5"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_gauss" \
  --which_optim="adam" --adam_weight_decay=0 \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_gauss_freeze" \
  --which_optim="adam" --adam_weight_decay=0 \
  --freeze_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_ortho" \
  --which_optim="adam" --adam_weight_decay=0 \
  --ortho_init_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_ortho_freeze" \
  --which_optim="adam" --adam_weight_decay=0 \
  --ortho_init_down \
  --freeze_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_gauss_minimax" \
  --which_optim="adam" --adam_weight_decay=0 \
  --how_to_optimize="minimax" \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/base_ortho_minimax" \
  --which_optim="adam" --adam_weight_decay=0 \
  --ortho_init_down \
  --how_to_optimize="minimax" \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/stiefel_gauss" \
  --which_optim="adam" --adam_weight_decay=0 \
  --stiefel_optim_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/stiefel_ortho" \
  --which_optim="adam" --adam_weight_decay=0 \
  --ortho_init_down \
  --stiefel_optim_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/stiefel_gauss_minimax" \
  --which_optim="adam" --adam_weight_decay=0 \
  --stiefel_optim_down \
  --how_to_optimize="minimax" \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/stiefel_ortho_minimax" \
  --which_optim="adam" --adam_weight_decay=0 \
  --ortho_init_down \
  --stiefel_optim_down \
  --how_to_optimize="minimax" \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/stiefel_gauss" \
  --which_optim="adam" --adam_weight_decay=0 \
  --stiefel_optim_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/sam_gauss" \
  --which_optim="adam" --adam_weight_decay=0 \
  --how_to_optimize="sam" \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/sam_ortho" \
  --which_optim="adam" --adam_weight_decay=0 \
  --how_to_optimize="sam" \
  --ortho_init_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 

CUDA_VISIBLE_DEVICES=$GPU_ID python3 training_scripts/train_rolora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/rolora_2/sam_ortho_freeze" \
  --which_optim="adam" --adam_weight_decay=0 \
  --how_to_optimize="sam" \
  --ortho_init_down \
  --freeze_down \
  --instance_prompt="photo of a sks dog" \
  --prompts="$PROMPTS" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate="${LR}" \
  --learning_rate_text="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 
