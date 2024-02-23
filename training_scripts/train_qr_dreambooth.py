# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import itertools
import math
import os,sys
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)
from lora_diffusion.lora import inject_trainable_kron_qr, inject_trainable_qr
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import DDIMScheduler
from utils_ext import image_grid

from pathlib import Path

import random
import re

from sam import SAM, disable_running_stats, enable_running_stats
from StiefelOptimizers import StiefelSGD, StiefelAdam


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        img_transforms = []

        if resize:
            img_transforms.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        required=True,
        help="sample prompts to generate images",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both","None"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    parser.add_argument("--ortho_init_up", action="store_true")
    parser.add_argument("--ortho_init_down", action="store_true")
    parser.add_argument("--zero_init_down", action="store_true")
    parser.add_argument("--triangle_down", action="store_true")
    parser.add_argument("--gaussianize_ortho_init_up", type=str, default='sphere')
    parser.add_argument("--gaussianize_ortho_init_down", type=str, default='sphere')
    parser.add_argument("--stiefel_optim_up", action="store_true")
    parser.add_argument("--stiefel_optim_down", action="store_true")
    parser.add_argument("--up_lr_multiplier", type=float, default=1.)
    parser.add_argument("--down_lr_multiplier", type=float, default=1.)
    # parser.add_argument("--maximize_down", action="store_true")
    parser.add_argument("--how_to_optimize", type=str, default="minimize", choices=["minimize", "minimax", "minimax2", "sam"])
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--sam_down_only", action="store_true")
    parser.add_argument("--freeze_down", action="store_true")
    parser.add_argument("--which_optim", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--factorization", type=str, default="qr", choices=["kron_qr", "lora","qr"])
    parser.add_argument("--stiefel_optim_kron_1", action="store_true")
    parser.add_argument("--stiefel_optim_kron_2", action="store_true")
    parser.add_argument("--stiefel_optim_qr_q", action="store_true")
    parser.add_argument("--stiefel_optim_qr_r", action="store_true")
    parser.add_argument("--kron_1_lr_multiplier", type=float, default=1.)
    parser.add_argument("--kron_2_lr_multiplier", type=float, default=1.)
    parser.add_argument("--qr_q_lr_multiplier", type=float, default=1.)
    parser.add_argument("--qr_r_lr_multiplier", type=float, default=1.)
    parser.add_argument("--size_kron_1", type=int, default=4)
    parser.add_argument("--size_qr_r", type=int, default=4)   

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            logger.warning(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args


def sample_images(args, unet, text_encoder, prompts, global_step, n_samples=1, seed=0):
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", eta=0.)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        scheduler=scheduler,
        revision=args.revision,
        safety_checker=None,
        feature_extractor=None,
    ).to('cuda')
    prompts = re.split("<SEP>|<sep>", prompts)
    for j, prompt in enumerate(prompts):
        prompt = [prompt] * n_samples
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5,
                      generator=torch.Generator('cuda').manual_seed(seed)).images
        grid = image_grid(images, rows=1, cols=n_samples)
        grid.save(f"{args.output_dir}/sample_{global_step}-{j}.png")


def p_loss(args, unet, text_encoder, latents, input_ids, noise_scheduler):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(input_ids)[0]

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    if args.with_prior_preservation:
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = (
            F.mse_loss(model_pred.float(), target.float(), reduction="none")
            .mean([1, 2, 3])
            .mean()
        )

        # Compute prior loss
        prior_loss = F.mse_loss(
            model_pred_prior.float(), target_prior.float(), reduction="mean"
        )

        # Add the prior loss to the instance loss.
        # loss = loss + args.prior_loss_weight * prior_loss
        return loss, prior_loss
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss, 0.

    # accelerator.backward(loss)
    # if accelerator.sync_gradients:
    #     params_to_clip = (
    #         itertools.chain(unet.parameters(), text_encoder.parameters())
    #         if args.train_text_encoder
    #         else unet.parameters()
    #     )
    #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet.requires_grad_(False)

    if args.factorization == 'lora':
        unet_lora_params, _ = inject_trainable_lora(
            unet, r=args.lora_rank, loras=args.resume_unet
        )
        unet_lora_params_up = itertools.islice(unet_lora_params, 0, None, 2)
        unet_lora_params_down = itertools.islice(unet_lora_params, 1, None, 2)

    if args.factorization == 'kron_qr':
        unet_kron_qr_params, _ = inject_trainable_kron_qr(
            unet, size_kron_1=args.size_kron_1
        )
        unet_kron_qr_params_kron_1 = itertools.islice(unet_kron_qr_params, 0, None, 3)
        unet_kron_qr_params_kron_2 = itertools.islice(unet_kron_qr_params, 1, None, 3)
        unet_kron_qr_params_qr_r = itertools.islice(unet_kron_qr_params, 2, None, 3)

    if args.factorization == 'lora':
        for _up, _down in extract_lora_ups_down(unet):
            print("Before training: Unet First Layer lora up", _up.weight.data)
            print("Before training: Unet First Layer lora down", _down.weight.data)
            break

    if args.factorization == 'qr':
        unet_qr_params, _ = inject_trainable_qr(
            unet,
        )
        unet_qr_params_qr_q = itertools.islice(unet_qr_params, 0, None, 2)
        unet_qr_params_qr_r = itertools.islice(unet_qr_params, 1, None, 2)   

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.train_text_encoder:
        if args.factorization == 'lora':
            text_encoder_lora_params, _ = inject_trainable_lora(
                text_encoder,
                target_replace_module=["CLIPAttention"],
                r=args.lora_rank,
            )
            text_encoder_lora_params_up = itertools.islice(text_encoder_lora_params, 0, None, 2)
            text_encoder_lora_params_down = itertools.islice(text_encoder_lora_params, 1, None, 2)
            for _up, _down in extract_lora_ups_down(
                text_encoder, target_replace_module=["CLIPAttention"]
            ):
                print("Before training: text encoder First Layer lora up", _up.weight.data)
                print(
                    "Before training: text encoder First Layer lora down", _down.weight.data
                )
                break
        
        if args.factorization == 'kron_qr':
            text_encoder_kron_qr_params, _ = inject_trainable_kron_qr(
                text_encoder,
                target_replace_module=["CLIPAttention"],
                size_kron_1=args.size_kron_1
            )
            text_encoder_kron_qr_params_kron_1 = itertools.islice(text_encoder_kron_qr_params, 0, None, 3)
            text_encoder_kron_qr_params_kron_2 = itertools.islice(text_encoder_kron_qr_params, 1, None, 3)
            text_encoder_kron_qr_params_qr_r = itertools.islice(text_encoder_kron_qr_params, 2, None, 3)       

        if args.factorization == 'qr':
            text_encoder_qr_params, _ = inject_trainable_qr(
                text_encoder,
                target_replace_module=["CLIPAttention"],
            )
            text_encoder_qr_params_qr_q = itertools.islice(text_encoder_qr_params, 0, None, 2)
            text_encoder_qr_params_qr_r = itertools.islice(text_encoder_qr_params, 1, None, 2)       

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    # if args.use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
    #         )

    #     optimizer_class = bnb.optim.AdamW8bit
    # else:
    #     optimizer_class = torch.optim.AdamW
    if args.which_optim == 'adam':
        optimizer_class = torch.optim.Adam
        optim_args = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.which_optim == 'adamw':
        optimizer_class = torch.optim.AdamW
        optim_args = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.which_optim == 'sgd':
        optimizer_class = torch.optim.SGD
        optim_args = dict(
            lr=args.learning_rate,
            momentum=args.adam_beta1,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.which_optim}")

    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )

    if args.factorization == 'kron_qr':
        # orthogonalize the kron1 weights
        unet_kron_qr_params_kron_1_list = list(itertools.chain(*unet_kron_qr_params_kron_1))
        text_encoder_kron_qr_params_kron_1_list = list(itertools.chain(*text_encoder_kron_qr_params_kron_1)) if args.train_text_encoder else []
        '''
        if args.ortho_init_kron_1:
            for param in unet_kron_qr_params_kron_1_list + text_encoder_kron_qr_params_kron_1_list:
                torch.nn.init.orthogonal_(param)
                if args.gaussianize_ortho_init_up == 'sphere':
                    d = max(param.shape[0], param.shape[1])
                    param.data = param.data * (1./args.lora_rank * math.sqrt(d))
            print("After orthogonalization")
        '''
        params_to_optimize_kron_1 = (
            [
                {"params": unet_kron_qr_params_kron_1_list, "lr": args.learning_rate * args.kron_1_lr_multiplier},
                {"params": text_encoder_kron_qr_params_kron_1_list, "lr": text_lr * args.kron_1_lr_multiplier},
            ]
            if args.train_text_encoder
            else unet_kron_qr_params_kron_1_list
        )

        optim_args['lr'] = args.learning_rate * args.kron_1_lr_multiplier

        if args.stiefel_optim_kron_1:
            if args.which_optim == 'adam':
                optimizer_kron_1 = StiefelAdam(
                    params_to_optimize_kron_1,
                    **optim_args,
                )
                base_optimizer = StiefelAdam
            elif args.which_optim == 'sgd':
                optimizer_kron_1 = StiefelSGD(
                    params_to_optimize_kron_1,
                    **optim_args,
                )
                base_optimizer = StiefelSGD
            else:
                raise ValueError(f"Unknown optimizer {args.which_optim}")
            print("After StiefelOptimizer")
        else:
            optimizer_kron_1 = optimizer_class(
                params_to_optimize_kron_1,
                **optim_args,
            )
            base_optimizer = optimizer_class
            print(f"Good old optimizer {args.which_optim}")
        if args.how_to_optimize == 'sam':
            optimizer_kron_1 = SAM(
                params_to_optimize_kron_1,
                base_optimizer,
                rho=args.sam_rho,
                **optim_args,
            )


        # orthogonalize the kron2 weights
        unet_kron_qr_params_kron_2_list = list(itertools.chain(*unet_kron_qr_params_kron_2))
        text_encoder_kron_qr_params_kron_2_list = list(itertools.chain(*text_encoder_kron_qr_params_kron_2)) if args.train_text_encoder else []
        '''
        if args.ortho_init_kron_2:
            for param in unet_kron_qr_params_kron_2_list + text_encoder_kron_qr_params_kron_2_list:
                torch.nn.init.orthogonal_(param)
                if args.gaussianize_ortho_init_up == 'sphere':
                    d = max(param.shape[0], param.shape[1])
                    param.data = param.data * (1./args.lora_rank * math.sqrt(d))
            print("After orthogonalization")
        '''
        params_to_optimize_kron_2 = (
            [
                {"params": unet_kron_qr_params_kron_2_list, "lr": args.learning_rate * args.kron_2_lr_multiplier},
                {"params": text_encoder_kron_qr_params_kron_2_list, "lr": text_lr * args.kron_2_lr_multiplier},
            ]
            if args.train_text_encoder
            else unet_kron_qr_params_kron_2_list
        )

        optim_args['lr'] = args.learning_rate * args.kron_2_lr_multiplier

        if args.stiefel_optim_kron_2:
            if args.which_optim == 'adam':
                optimizer_kron_2 = StiefelAdam(
                    params_to_optimize_kron_2,
                    **optim_args,
                )
                base_optimizer = StiefelAdam
            elif args.which_optim == 'sgd':
                optimizer_kron_2 = StiefelSGD(
                    params_to_optimize_kron_2,
                    **optim_args,
                )
                base_optimizer = StiefelSGD
            else:
                raise ValueError(f"Unknown optimizer {args.which_optim}")
            print("After StiefelOptimizer")
        else:
            optimizer_kron_2 = optimizer_class(
                params_to_optimize_kron_2,
                **optim_args,
            )
            base_optimizer = optimizer_class
            print(f"Good old optimizer {args.which_optim}")
        if args.how_to_optimize == 'sam':
            optimizer_kron_2 = SAM(
                params_to_optimize_kron_2,
                base_optimizer,
                rho=args.sam_rho,
                **optim_args,
            )
            

        # orthogonalize the qr_r weights
        unet_kron_qr_params_qr_r_list = list(itertools.chain(*unet_kron_qr_params_qr_r))
        text_encoder_kron_qr_params_qr_r_list = list(itertools.chain(*text_encoder_kron_qr_params_qr_r)) if args.train_text_encoder else []
        '''
        if args.ortho_init_qr_r:
            for param in unet_kron_qr_params_qr_r_list + text_encoder_kron_qr_params_qr_r_list:
                torch.nn.init.orthogonal_(param)
                if args.gaussianize_ortho_init_up == 'sphere':
                    d = max(param.shape[0], param.shape[1])
                    param.data = param.data * (1./args.lora_rank * math.sqrt(d))
            print("After orthogonalization")
        '''
        params_to_optimize_qr_r = (
            [
                {"params": unet_kron_qr_params_qr_r_list, "lr": args.learning_rate * args.qr_r_lr_multiplier},
                {"params": text_encoder_kron_qr_params_qr_r_list, "lr": text_lr * args.qr_r_lr_multiplier},
            ]
            if args.train_text_encoder
            else unet_kron_qr_params_qr_r_list
        )

        optim_args['lr'] = args.learning_rate * args.qr_r_lr_multiplier

        if args.stiefel_optim_qr_r:
            if args.which_optim == 'adam':
                optimizer_qr_r = StiefelAdam(
                    params_to_optimize_qr_r,
                    **optim_args,
                )
                base_optimizer = StiefelAdam
            elif args.which_optim == 'sgd':
                optimizer_qr_r = StiefelSGD(
                    params_to_optimize_qr_r,
                    **optim_args,
                )
                base_optimizer = StiefelSGD
            else:
                raise ValueError(f"Unknown optimizer {args.which_optim}")
            print("After StiefelOptimizer")
        else:
            optimizer_qr_r = optimizer_class(
                params_to_optimize_qr_r,
                **optim_args,
            )
            base_optimizer = optimizer_class
            print(f"Good old optimizer {args.which_optim}")
        if args.how_to_optimize == 'sam':
            optimizer_qr_r = SAM(
                params_to_optimize_qr_r,
                base_optimizer,
                rho=args.sam_rho,
                **optim_args,
            )

        
        print(optimizer_kron_1)
        print(optimizer_kron_2)
        print(optimizer_qr_r)

    if args.factorization == 'qr':
        # orthogonalize the qr_q weights
        unet_qr_params_qr_q_list = list(itertools.chain(*unet_qr_params_qr_q))
        text_encoder_qr_params_qr_q_list = list(itertools.chain(*text_encoder_qr_params_qr_q)) if args.train_text_encoder else []
        '''
        if args.ortho_init_qr_r:
            for param in unet_kron_qr_params_qr_r_list + text_encoder_kron_qr_params_qr_r_list:
                torch.nn.init.orthogonal_(param)
                if args.gaussianize_ortho_init_up == 'sphere':
                    d = max(param.shape[0], param.shape[1])
                    param.data = param.data * (1./args.lora_rank * math.sqrt(d))
            print("After orthogonalization")
        '''
        params_to_optimize_qr_q = (
            [
                {"params": unet_qr_params_qr_q_list, "lr": args.learning_rate * args.qr_q_lr_multiplier},
                {"params": text_encoder_qr_params_qr_q_list, "lr": text_lr * args.qr_q_lr_multiplier},
            ]
            if args.train_text_encoder
            else text_encoder_qr_params_qr_q_list
        )

        optim_args['lr'] = args.learning_rate * args.qr_q_lr_multiplier

        if args.stiefel_optim_qr_q:
            if args.which_optim == 'adam':
                optimizer_qr_q = StiefelAdam(
                    params_to_optimize_qr_q,
                    **optim_args,
                )
                base_optimizer = StiefelAdam
            elif args.which_optim == 'sgd':
                optimizer_qr_q = StiefelSGD(
                    params_to_optimize_qr_q,
                    **optim_args,
                )
                base_optimizer = StiefelSGD
            else:
                raise ValueError(f"Unknown optimizer {args.which_optim}")
            print("After StiefelOptimizer")
        else:
            optimizer_qr_q = optimizer_class(
                params_to_optimize_qr_q,
                **optim_args,
            )
            base_optimizer = optimizer_class
            print(f"Good old optimizer {args.which_optim}")
        if args.how_to_optimize == 'sam':
            optimizer_qr_q = SAM(
                params_to_optimize_qr_q,
                base_optimizer,
                rho=args.sam_rho,
                **optim_args,
            )

        # orthogonalize the qr_r weights
        unet_qr_params_qr_r_list = list(itertools.chain(*unet_qr_params_qr_r))
        text_encoder_qr_params_qr_r_list = list(itertools.chain(*text_encoder_qr_params_qr_r)) if args.train_text_encoder else []
        '''
        if args.ortho_init_qr_r:
            for param in unet_kron_qr_params_qr_r_list + text_encoder_kron_qr_params_qr_r_list:
                torch.nn.init.orthogonal_(param)
                if args.gaussianize_ortho_init_up == 'sphere':
                    d = max(param.shape[0], param.shape[1])
                    param.data = param.data * (1./args.lora_rank * math.sqrt(d))
            print("After orthogonalization")
        '''
        params_to_optimize_qr_r = (
            [
                {"params": unet_qr_params_qr_r_list, "lr": args.learning_rate * args.qr_r_lr_multiplier},
                {"params": text_encoder_qr_params_qr_r_list, "lr": text_lr * args.qr_r_lr_multiplier},
            ]
            if args.train_text_encoder
            else unet_qr_params_qr_r_list
        )

        optim_args['lr'] = args.learning_rate * args.qr_r_lr_multiplier

        if args.stiefel_optim_qr_r:
            if args.which_optim == 'adam':
                optimizer_qr_r = StiefelAdam(
                    params_to_optimize_qr_r,
                    **optim_args,
                )
                base_optimizer = StiefelAdam
            elif args.which_optim == 'sgd':
                optimizer_qr_r = StiefelSGD(
                    params_to_optimize_qr_r,
                    **optim_args,
                )
                base_optimizer = StiefelSGD
            else:
                raise ValueError(f"Unknown optimizer {args.which_optim}")
            print("After StiefelOptimizer")
        else:
            optimizer_qr_r = optimizer_class(
                params_to_optimize_qr_r,
                **optim_args,
            )
            base_optimizer = optimizer_class
            print(f"Good old optimizer {args.which_optim}")
        if args.how_to_optimize == 'sam':
            optimizer_qr_r = SAM(
                params_to_optimize_qr_r,
                base_optimizer,
                rho=args.sam_rho,
                **optim_args,
            )    

        print(optimizer_qr_q)
        print(optimizer_qr_r)

    if args.factorization == 'lora':
        for _up, _down in extract_lora_ups_down(unet):
            print("After init: Unet First Layer lora up", _up.weight.data)
            print("After init: Unet First Layer lora down", _down.weight.data)
            break

        if args.train_text_encoder:
            for _up, _down in extract_lora_ups_down(
                text_encoder, target_replace_module=["CLIPAttention"]
            ):
                print("After init: text encoder First Layer lora up", _up.weight.data)
                print(
                    "After init: text encoder First Layer lora down", _down.weight.data
                )
                break

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.factorization == 'kron_qr':
        lr_scheduler_kron_1 = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_kron_1,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        lr_scheduler_kron_2 = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_kron_2,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        lr_scheduler_qr_r = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_qr_r,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    if args.factorization == 'qr':
        lr_scheduler_qr_q = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_qr_q,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )       

        lr_scheduler_qr_r = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_qr_r,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    if args.factorization == 'kron_qr':
        if args.train_text_encoder:
            (
                unet,
                text_encoder,
                optimizer_kron_1, optimizer_kron_2, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_kron_1, lr_scheduler_kron_2, lr_scheduler_qr_r,
            ) = accelerator.prepare(
                unet,
                text_encoder,
                optimizer_kron_1, optimizer_kron_2, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_kron_1, lr_scheduler_kron_2, lr_scheduler_qr_r
            )
        else:
            (
                unet,
                optimizer_kron_1, optimizer_kron_2, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_kron_1, lr_scheduler_kron_2, lr_scheduler_qr_r
            ) = accelerator.prepare(
                unet,
                optimizer_kron_1, optimizer_kron_2, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_kron_1, lr_scheduler_kron_2, lr_scheduler_qr_r
            )

    if args.factorization == 'qr':
        if args.train_text_encoder:
            (
                unet,
                text_encoder,
                optimizer_qr_q, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_qr_q, lr_scheduler_qr_r,
            ) = accelerator.prepare(
                unet,
                text_encoder,
                optimizer_qr_q, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_qr_q, lr_scheduler_qr_r,
            )
        else:
            (
                unet,
                optimizer_qr_q, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_qr_q, lr_scheduler_qr_r,
            ) = accelerator.prepare(
                unet,
                optimizer_qr_q, optimizer_qr_r,
                train_dataloader,
                lr_scheduler_qr_q, lr_scheduler_qr_r,
            )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215


            if args.how_to_optimize == "minimize":
                loss, prior_loss = p_loss(args, unet, text_encoder, latents, batch["input_ids"], noise_scheduler)
                # accelerator.backward(loss + args.prior_loss_weight * prior_loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if args.factorization == 'kron_qr':
                    optimizer_kron_1.step()
                    optimizer_kron_2.step()
                    optimizer_qr_r.step()
                if args.factorization == 'qr':
                    optimizer_qr_q.step()
                    optimizer_qr_r.step()

                if args.triangle_down:
                    for param in unet_qr_params_qr_r_list + text_encoder_qr_params_qr_r_list:
                        with torch.no_grad():
                            upper_band_mask = torch.triu(torch.ones_like(param), diagonal=0)
                            lower_band_mask = torch.tril(torch.ones_like(param), diagonal=args.size_qr_r)
                            param.copy_(param * upper_band_mask * lower_band_mask)
                print(param)

                if args.factorization == 'kron_qr':
                    optimizer_kron_1.zero_grad()
                    optimizer_kron_2.zero_grad()
                    optimizer_qr_r.zero_grad()
                if args.factorization == 'qr':
                    optimizer_qr_q.zero_grad()
                    optimizer_qr_r.zero_grad()                                   

            if args.factorization == 'kron_qr':
                lr_scheduler_kron_1.step()
                lr_scheduler_kron_2.step()
                lr_scheduler_qr_r.step()
            if args.factorization == 'qr':
                lr_scheduler_qr_q.step()
                lr_scheduler_qr_r.step()

            progress_bar.update(1)
            global_step += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            '''
            if accelerator.sync_gradients:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                        # it, the models will be unwrapped, and when they are then used for further training,
                        # we will crash. pass this, but only to newer versions of accelerate. fixes
                        # https://github.com/huggingface/diffusers/issues/1566
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True}
                            if accepts_keep_fp32_wrapper
                            else {}
                        )
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, **extra_args
                            ),
                            revision=args.revision,
                        )
                        
                        filename_unet = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                        )
                        filename_text_encoder = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                        print(f"save weights {filename_unet}, {filename_text_encoder}")
                        save_lora_weight(pipeline.unet, filename_unet)
                        if args.train_text_encoder:
                            save_lora_weight(
                                pipeline.text_encoder,
                                filename_text_encoder,
                                target_replace_module=["CLIPAttention"],
                            )

                        for _up, _down in extract_lora_ups_down(pipeline.unet):
                            print(
                                "First Unet Layer's Up Weight is now : ",
                                _up.weight.data, _up.weight.data.size(),
                            )
                            print(
                                "First Unet Layer's Down Weight is now : ",
                                _down.weight.data, _down.weight.data.size(),
                            )
                            break
                        if args.train_text_encoder:
                            for _up, _down in extract_lora_ups_down(
                                pipeline.text_encoder,
                                target_replace_module=["CLIPAttention"],
                            ):
                                print(
                                    "First Text Encoder Layer's Up Weight is now : ",
                                    _up.weight.data,
                                )
                                print(
                                    "First Text Encoder Layer's Down Weight is now : ",
                                    _down.weight.data,
                                )
                                break

                        last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler_up.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            '''

    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\nLora TRAINING DONE!\n\n")

        # sample images
        if args.prompts is not None:
            sample_images(
                args,
                accelerator.unwrap_model(unet),
                accelerator.unwrap_model(text_encoder),
                args.prompts,
                global_step,
                n_samples=4,
                seed=0,
            )

        if args.output_format == "pt" or args.output_format == "both":
            save_lora_weight(pipeline.unet, args.output_dir + "/lora_weight.pt")
            if args.train_text_encoder:
                save_lora_weight(
                    pipeline.text_encoder,
                    args.output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )

        if args.output_format == "safe" or args.output_format == "both":
            loras = {}
            loras["unet"] = (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"})
            if args.train_text_encoder:
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            save_safeloras(loras, args.output_dir + "/lora_weight.safetensors")

        # if args.push_to_hub:
        #     repo.push_to_hub(
        #         commit_message="End of training",
        #         blocking=False,
        #         auto_lfs_prune=True,
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)