#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="./output_example"
export CUDA_VISIBLE_DEVICES=4,5,6,7
#export NCCL_DEBUG=INFO
#export TORCH_SHOW_CPP_STACKTRACES=1

accelerate launch training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="style of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000