import re
import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch


""" image utilities
"""
# NOTE: copied from https://huggingface.co/blog/stable_diffusion
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def split_negative_prompt(prompt, sep_neg="<NEG>|<neg>"):
    prompt, *negative_prompt = re.split(sep_neg, prompt)
    # negative_prompt = sep_neg.split('|')[0].join([s.strip() for s in negative_prompt if s])
    negative_prompt = sep_neg.split('|')[0].join([s.strip() for s in negative_prompt])
    negative_prompt = negative_prompt if negative_prompt else None
    return prompt, negative_prompt


def split_prompt(prompt, sep="<SEP>|<sep>", sep_neg="<NEG>|<neg>"):
    prompts = []
    negative_prompts = []
    for p in re.split(sep, prompt):
        p, np = split_negative_prompt(p, sep_neg=sep_neg)
        prompts.append(p)
        negative_prompts.append(np)
    return prompts, negative_prompts


def text_to_image_grid(
    pipe,
    prompt,
    n_samples=1,
    guidance_scale=7.5,
):
    """ prompt is separated by <sep>
    """
    grids = []
    for j, p in enumerate(re.split("<SEP>|<sep>", prompt)):
        if bool(re.search("<NEG>|<neg>", p)):
            p, np = re.split("<NEG>|<neg>", p)
        else:
            np = None
        p = [p] * n_samples
        np = [np] * n_samples if np is not None else None
        images = pipe(prompt=p, negative_prompt=np,
            num_inference_steps=100, guidance_scale=guidance_scale).images
        grid = image_grid(images, rows=1, cols=n_samples)
        grids.append(grid)
    return grids


""" logging utilities
"""

def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except:
        return 'unknown'


def print_args(parser, args, logdir="logs", name=None, is_dict=False, flush=False, backup_filelist=[]):
    args = deepcopy(args)
    if not is_dict and hasattr(args, 'parser'):
        delattr(args, 'parser')
    if name is None:
        name = Path(logdir).name
    datetime_now = datetime.now()
    message = f"Name: {name} Time: {datetime_now}\n"
    message += f"{os.getenv('USER')}@{get_hostname()}:\n"
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        message += f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}\n"
    message += '--------------- Arguments ---------------\n'
    args_vars = args if is_dict else vars(args)
    for k, v in sorted(args_vars.items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    if flush:
        print(message)

    # save to the disk
    logdir = Path(logdir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir / 'src', exist_ok=True)
    filename = logdir / 'src' / 'args.txt'
    with open(filename, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    sys_argv = deepcopy(sys.argv)
    command = [sys_argv[0]]
    for arg in sys_argv[1:]:
        command.append(f'"{arg}"' if ' ' in arg else arg)

    with open(logdir / 'src' / 'cmd.txt', 'a+') as f:
        f.write(f'# Time: {datetime_now}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(command))
        f.write('\n\n')

    # backup source files
    shutil.copyfile(sys.argv[0], logdir / 'src' / f'{os.path.basename(sys.argv[0])}.txt')
    if isinstance(backup_filelist, str):
        backup_filelist = re.split(",|:|;", backup_filelist)
    for filepath in backup_filelist:
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(logdir, 'src', filename+'.txt'))


def find_ckpt(ckpt_dir, file_name, extentions=['.ckpt', '.bin', '.pt', '.pth']):
    for name in os.listdir(ckpt_dir):
        if name.startswith(file_name):
            for ext in extentions:
                if name.endswith(ext):
                    return os.path.join(ckpt_dir, name)
    return None


""" misc utilities
"""
def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)
