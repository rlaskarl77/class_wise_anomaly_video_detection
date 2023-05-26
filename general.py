import logging
import logging.config
import platform
import os
import pkg_resources as pkg
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


LOGGING_NAME = "vidan"


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)


def increment_path(path, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def select_device(device='', batch_size=0, newline=True):
    s=''
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'
    if not newline:
        s = s.rstrip()
        
    LOGGER.info(s)
    
    return torch.device(arg)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def smart_resume(ckpt, optimizer, weights='yolov5s.pt', epochs=300, resume=True):
    # Resume training from a partially trained checkpoint
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
        best_fitness = ckpt['best_fitness']
    if resume:
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.\n' \
                                f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        LOGGER.info(f'Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs')
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt['epoch']  # finetune additional epochs
        
    return best_fitness, start_epoch, epochs


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if verbose and not result:
        LOGGER.warning(s)
    return result


def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    

def smart_resume(ckpt, optimizer, ema=None, epochs=300, resume=True):
    # Resume training from a partially trained checkpoint
    best_auc = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
        best_auc = ckpt['best_auc']
    if resume:
        assert start_epoch > 0, f'{start_epoch} should not be zero (resuming)'
        LOGGER.info(f'Resuming training from epoch {start_epoch} to {epochs} total epochs')
    if epochs < start_epoch:
        LOGGER.info(f"Current checkpoint is trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt['epoch']  # finetune additional epochs
    return best_auc, start_epoch, epochs