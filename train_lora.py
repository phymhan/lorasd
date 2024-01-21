import argparse
import logging
import hashlib
import inspect
import itertools
import math
import os
import re
from pathlib import Path
from typing import Optional
from ast import literal_eval
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import diffusers
# import PIL
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils_svd import register_spectral_shift, register_low_rank
from utils_ext import print_args, image_grid, split_prompt
from pipeline import StableDiffusionPipelineDev


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

dummy_safety_checker = None


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
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
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=-1,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt. If -1, all images in class_data_dir."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/svdiffusion_experiment",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=str, default='500', help="Save checkpoint every X updates steps.")
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
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--enable_debug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)

    # Arguments for SVDiff
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to use for inference.")
    parser.add_argument("--parameter_space", type=str, default="full")
    parser.add_argument("--convert_svd_to_full", action="store_true", help="Whether to convert SVD weights to full weights (and save).")
    parser.add_argument("--ckpt_svd_path", type=str, default=None)
    parser.add_argument("--bias_lr_multiplier", type=float, default=0,
        help="actual lr for bias is base_learning_rate * bias_lr_multiplier")
    parser.add_argument("--weight1d_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--weight2d_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--weight3d_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--weight4d_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--save_initial_weights", action="store_true", help="Whether to save initial weights.")
    parser.add_argument("--svd_rank", type=int, default=None, help="Rank of SVD, if None, use full rank.")
    parser.add_argument("--svd_shift_init_scale", type=float, default=0.0, help="Scale of the shift initialization.")
    parser.add_argument("--svd_no_residual", action="store_true", help="Whether to add residual in SVD reconstruction.")
    
    # Arguments for personalized dataset
    parser.add_argument("--which_dataset", type=str, default="dreambooth", choices=["dreambooth", "personalized", "personalized_multi", "single_image"])
    parser.add_argument("--cropping", type=str, default="random", choices=["center", "random"])
    parser.add_argument("--crop_scale", type=str, default="1,1")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--placeholder_token", type=str, default="sks", help="only used for personalized_multi or cutmix dataset")
    parser.add_argument("--coarse_class_text", type=str, default=None, help="only used when cutmix_prob > 0")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--which_scheduler", type=str, default="ddim")
    parser.add_argument("--skip_saving", action="store_true", help="Whether to skip saving the model.")
    parser.add_argument("--legacy", action="store_true", help="Whether to use legacy code.")
    parser.add_argument("--width_scale", type=str, default="0.5,0.5")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per prompt.")
    parser.add_argument("--instance_sample_weights", type=str, default=None, help="sample instances with these probabilities")
    parser.add_argument("--class_sample_weights", type=str, default=None, help="sample classes with these probabilities")
    parser.add_argument("--no_resample_cutmix", action="store_true")
    parser.add_argument("--cutmix_prob", type=float, default=0., help="only used for cutmix dataset; if 0, equivalent to multi datasets")
    parser.add_argument("--cutmix_and_prompt_prob", type=float, default=0.5, help="probability of use 'and' prompt rather than 'left and right' prompt")
    parser.add_argument("--cutmix_ordered_keys", type=int, default=0)
    parser.add_argument("--cutmix_allow_same_keys", type=int, default=0)
    parser.add_argument("--cutmix_number_prompt_prob", type=float, default=0)
    parser.add_argument("--cutmix_style_prob", type=float, default=0, help="add `cutmix style' to prompt")
    parser.add_argument("--class_cutmix_prob", type=float, default=None, help="only used for cutmix dataset; if 0, equivalent to multi datasets")
    parser.add_argument("--class_cutmix_and_prompt_prob", type=float, default=None, help="probability of use 'and' prompt rather than 'left and right' prompt")
    parser.add_argument("--class_cutmix_ordered_keys", type=int, default=None)
    parser.add_argument("--class_cutmix_allow_same_keys", type=int, default=None)
    parser.add_argument("--class_cutmix_number_prompt_prob", type=float, default=None)
    parser.add_argument("--class_cutmix_style_prob", type=float, default=None, help="add `cutmix style' to prompt")
    parser.add_argument("--cutmix_change_after_step", type=int, default=None, help="change cutmix probability after this many steps")
    parser.add_argument("--cutmix_prob2", type=float, default=0., help="only used for cutmix dataset; if 0, equivalent to multi datasets")
    parser.add_argument("--class_cutmix_prob2", type=float, default=0., help="only used for cutmix dataset; if 0, equivalent to multi datasets")
    parser.add_argument("--cutmix_prompt", type=str, default=None)
    parser.add_argument("--class_cutmix_prompt", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--save_intermediate_checkpoints", action="store_true", help="Whether to save intermediate full-weight checkpoints.")

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
    # else:
    #     if args.class_data_dir is not None:
    #         logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
    #     if args.class_prompt is not None:
    #         logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return parser, args


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
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:  # NOTE: use prior preservation loss
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root is not None:  # NOTE: use prior preservation loss
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


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


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def do_save(current_step, save_steps):
    if isinstance(save_steps, int):
        return current_step % save_steps == 0
    else:
        return current_step in save_steps


def get_delta_state_dict(model):
    state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if 'lora_' in k:
            new_state_dict[k] = state_dict[k]
    return new_state_dict


def save_checkpoint(args, unet, text_encoder, global_step, save_delta_only=False, sample_image=False, seed=None):
    # NOTE: assumes models are already unwraped
    pipeline = None
    if args.svdiff and save_delta_only:
        # NOTE: save SVD deltas
        os.makedirs(os.path.join(args.output_dir, f"checkpoint-{global_step}"), exist_ok=True)
        torch.save({'state_dict': get_delta_state_dict(unet)}, 
            os.path.join(args.output_dir, f"checkpoint-{global_step}", "unet_delta.bin"))
        if args.train_text_encoder:
            torch.save({'state_dict': get_delta_state_dict(text_encoder)}, 
                os.path.join(args.output_dir, f"checkpoint-{global_step}", "text_encoder_delta.bin"))
    else:
        pipeline = StableDiffusionPipelineDev.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            safety_checker=None,
            feature_extractor=None,
            revision=args.revision,
        )
        if not args.skip_saving and (
            args.save_intermediate_checkpoints or (isinstance(global_step, str) and global_step == "last")
        ):
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            pipeline.save_pretrained(save_path)

    if sample_image:
        scheduler = None
        if args.which_scheduler == 'ddim':
            from diffusers import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", eta=0.)
        else:
            raise NotImplementedError
        if pipeline is None:
            pipeline = StableDiffusionPipelineDev.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                text_encoder=text_encoder,
                safety_checker=None,
                feature_extractor=None,
                revision=args.revision,
            ).to('cuda')
        else:
            pipeline.to('cuda')
        if scheduler is not None:
            pipeline.scheduler = scheduler
        os.makedirs(os.path.join(args.output_dir, f"sample"), exist_ok=True)
        prompts, negative_prompts = split_prompt(args.prompt, sep="<SEP>|<sep>", sep_neg="<NEG>|<neg>")
        for j, (prompt, negative_prompt) in enumerate(zip(prompts, negative_prompts)):
            prompt = [prompt] * args.n_samples
            negative_prompt = [negative_prompt] * args.n_samples if negative_prompt is not None else None
            image = pipeline(prompt, negative_prompt=negative_prompt,
                num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed) if seed is not None else None,
            ).images
            grid = image_grid(image, rows=1, cols=args.n_samples)
            grid.save(os.path.join(args.output_dir, f"sample", f"{global_step}-{j}.png"))


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)
    
    # NOTE: process some args
    if args.prompt is None:
        args.prompt = args.instance_prompt
    # NOTE: reload svd as lora
    args.svdiff = args.parameter_space in ["svd", "weight_svd", "lora"]

    if args.with_prior_preservation and args.num_class_images > 0:
        for class_data_dir, class_prompt in zip(re.split(",|:|;", args.class_data_dir), re.split("<SEP>|<sep>", args.class_prompt)):
            class_images_dir = Path(class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))
            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")
                sample_dataset = PromptDataset(class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images
                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if args.save_initial_weights and accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=dummy_safety_checker,
            revision=args.revision,
        )
        save_path = os.path.join(args.output_dir, f"checkpoint-initial")
        pipeline.save_pretrained(save_path)

    # NOTE: Wrap models with SVD parametrization if needed
    new_module_params = None
    new_module_model = None
    bias_trainable = False
    if args.svdiff:
        svd_kwargs = {}
        args_dict = vars(args)
        for k in args_dict:
            if k.startswith('svd_') or k.startswith('weight_'):
                svd_kwargs[k] = args_dict[k]
        bias_trainable = args.bias_lr_multiplier > 0
        parametrized_module_list = []
        new_module_dict = {'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []}
        new_module_params = {'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []}

        register_low_rank(
            unet,
            rank=args.lora_rank,
            bias_trainable=bias_trainable,
            svd_kwargs=svd_kwargs,
            parametrized_module_list=parametrized_module_list,
            new_module_dict=new_module_dict,
            new_module_params=new_module_params,
        )
        if args.train_text_encoder:
            register_low_rank(
                text_encoder,
                rank=args.lora_rank,
                bias_trainable=bias_trainable,
                svd_kwargs=svd_kwargs,
                parametrized_module_list=parametrized_module_list,
                new_module_dict=new_module_dict,
                new_module_params=new_module_params,
            )
        new_module_model = torch.nn.ModuleDict({k: torch.nn.ModuleList(v) for k, v in new_module_dict.items()})

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.svdiff:
        learning_rates = {
            '1d': args.learning_rate * args.weight1d_lr_multiplier,
            '2d': args.learning_rate * args.weight2d_lr_multiplier,
            '3d': args.learning_rate * args.weight3d_lr_multiplier,
            '4d': args.learning_rate * args.weight4d_lr_multiplier,
            'bias': args.learning_rate * args.bias_lr_multiplier,
        }
        params_to_optimize = {
            k: new_module_model[k].parameters() for k in new_module_model.keys() if (learning_rates[k] > 0)
        }
        optimizer = optimizer_class([
            {'params': params_to_optimize[k], 'lr': learning_rates[k]} for k in params_to_optimize.keys()],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
        )
        if args.legacy:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=args.learning_rate,
            )
        else:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

    # NOTE: use original DDPM scheduler for finetuning
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.which_dataset == 'dreambooth':
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )
    elif args.which_dataset == 'personalized':
        from utils_data import PersonalizedBase
        train_dataset = PersonalizedBase(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            cropping=args.cropping,
            crop_scale=args.crop_scale,
            flip_prob=args.flip_prob,
        )
    elif args.which_dataset == 'personalized_multi':
        cutmix_kwargs = {}
        args_dict = vars(args)
        for k, v in args_dict.items():
            if k.startswith('cutmix_') or k.startswith('class_cutmix_'):
                cutmix_kwargs[k] = v
        from utils_data import PersonalizedMulti
        train_dataset = PersonalizedMulti(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            cropping=args.cropping,
            crop_scale=args.crop_scale,
            flip_prob=args.flip_prob,
            placeholder_token=args.placeholder_token,
            coarse_class_text=args.coarse_class_text,
            legacy=args.legacy,
            width_scale=args.width_scale,
            instance_sample_weights=args.instance_sample_weights,
            class_sample_weights=args.class_sample_weights,
            resample_cutmix=not args.no_resample_cutmix,
            **cutmix_kwargs,
        )
    elif args.which_dataset == 'single_image':
        from utils_data import SingleImageBase
        train_dataset = SingleImageBase(
            instance_data_path=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            cropping=args.cropping,
            crop_scale=args.crop_scale,
            flip_prob=args.flip_prob,
        )
    else:
        raise NotImplementedError(f"Dataset {args.which_dataset} not implemented.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=0 if args.enable_debug else args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    if new_module_model is not None:
        new_module_model = accelerator.prepare(new_module_model)

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

    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unet.dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {unet.dtype}. {low_precision_error_string}")

    if args.train_text_encoder and text_encoder.dtype != torch.float32:
        raise ValueError(f"Text encoder loaded as datatype {text_encoder.dtype}. {low_precision_error_string}")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
    
    # NOTE: preprocess some args
    if isinstance(args.save_steps, str):
        args.save_steps = literal_eval(args.save_steps)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # NOTE: added as a temporary feature
            if args.cutmix_change_after_step is not None and global_step > args.cutmix_change_after_step:
                train_dataset.cutmix_prob = args.cutmix_prob2
                train_dataset.class_cutmix_prob = args.class_cutmix_prob2
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if do_save(global_step, args.save_steps):
                    if accelerator.is_main_process:
                        # When 'keep_fp32_wrapper' is `False` (the default), then the models are
                        # unwrapped and the mixed precision hooks are removed, so training crashes
                        # when the unwrapped models are used for further training.
                        # This is only supported in newer versions of `accelerate`.
                        # TODO(Pedro, Suraj): Remove `accepts_keep_fp32_wrapper` when forcing newer accelerate versions
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(accelerator.unwrap_model).parameters.keys()
                        )  # NOTE: True
                        extra_args = {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                        # NOTE: pass a random seed and use generator so that sampling won't change the random state of training
                        save_checkpoint(args,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                            global_step=global_step, sample_image=True,
                            seed=args.seed+global_step if args.seed is not None else global_step,
                            save_delta_only=True)  # NOTE: always save delta when using svdiff

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        if args.svdiff and args.convert_svd_to_full: # NOTE: only in main process?
            for layer in parametrized_module_list:
                torch.nn.utils.parametrize.remove_parametrizations(layer, "weight")
        extra_args = {}
        save_checkpoint(args,
            unet=accelerator.unwrap_model(unet, **extra_args),
            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
            global_step='last', sample_image=True,
            seed=args.seed+global_step if args.seed is not None else global_step,
            save_delta_only=not args.convert_svd_to_full)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    parser, args = parse_args()
    print_args(parser, args,
        logdir=args.output_dir,
        backup_filelist=[s for s in os.listdir('./') if s.endswith('.py')],
    )
    main(args)
