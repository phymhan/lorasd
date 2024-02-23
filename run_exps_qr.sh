#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="./output_example"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OUTPUT_DIR="./outputs/debug"
# export CLASS_DATA_DIR="./images_class/dog/samples"
PROMPTS="photo of a sks dog<SEP>photo of a sks dog in Times Square<SEP>a sks dog swimming pool<SEP>a sks dog made of Lego blocks<SEP>a sks dog in Minecraft style<SEP>watercolor painting of a sks dog"

LR="1e-5"
#--main_process_port 29501
accelerate launch training_scripts/train_qr_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir="logs/qr_2/qr_lora_band4_lr1_lr1" \
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
  --max_train_steps=1000 \
  --lora_rank=10 \
  --output_format="None" \
  --stiefel_optim_qr_q \
  --qr_q_lr_multiplier=1 \
  --qr_r_lr_multiplier=1 \
  --size_qr_r=4 \
  --triangle_down \
  #--up_lr_multiplier=500
  # -size_kron_1=16 \
