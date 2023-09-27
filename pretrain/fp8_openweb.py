"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import csv
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Config
from lit_gpt.utils import chunked_cross_entropy

# Float8 imports
from float8_experimental.float8_linear import (
    swap_linear_with_float8_linear, sync_float8_amax_and_scale_history)
from float8_experimental.float8_linear_nots import \
    swap_linear_with_float8_linear_nots


instruction_tuning = True
eval_interval = 25
save_interval = 10000
eval_iters = 100
log_interval = 1
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 6e-4
batch_size = 128
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 600000  # train dataset size
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5
model_name = "Llama-2-7b-hf"
name = "openwebtext"
out_dir = Path("out") / name
data_dir = Path("data") / name

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

device = torch.device("cuda")
val_step_count=0

# We want to skip the first embedding layer since scaled_mm needs to multiple of 16
float8_skip_list = ["lm_head"]
USE_TS = True

# OVERFIT TEST
OVERFIT=False


def write_loss_to_file(loss_file: Path, step: int, loss: float):
    if not loss_file.exists():
        with open(loss_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
    with open(loss_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([step, loss])


def get_profile_context(profile: bool, use_fp8: bool):
    def trace_handler(prof):
        fp8_str = "fp8_TS" if USE_TS else "fp8_NoTS"
        dtype_str = fp8_str if use_fp8 else "bf16"
        output_str = f"/tmp/trace_llama_7b_hf_{dtype_str}.json"
        prof.export_chrome_trace(output_str)
        print(f"Wrote profile to: {output_str}")
    if profile:
        context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=100,
            warmup=1,
            active=2,
            repeat=1),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler
        )
        return context
    else:
        return nullcontext()


def main(
    compile: bool = False,
    use_fp8: bool = False,
    profile: bool = False, # this will profile iterations 100-105
):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    os.makedirs(out_dir, exist_ok=True)

    config = Config.from_name(model_name)

    print("Initializing the model")
    with device:
        model = GPT(config).to(torch.bfloat16)
        model.apply(model._init_weights)

    print("setting up the dataloaders")
    train_data, val_data = load_datasets(data_dir, max_seq_length=model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=2)

    if use_fp8:
        if USE_TS:
            swap_linear_with_float8_linear(model)
        else:
            swap_linear_with_float8_linear_nots(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"The number of trainable parameters: {num_trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    if compile:
        model = torch.compile(model)

    train(model, optimizer, train_dataloader, val_dataloader, out_dir, use_fp8, profile)

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_full_finetuned.pth"
    torch.save(save_path, {"model": model})

def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    out_dir: str,
    use_fp8: bool,
    profile: bool,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    total_lengths = 0
    progress_bar = tqdm(total=max_iters)

    model.train()
    profile_context = get_profile_context(profile, use_fp8)
    train_iter = iter(train_data)

    # Sanity check
    fp8_str = "fp8_TS" if USE_TS else "fp8_NoTS"
    dtype_str = fp8_str if use_fp8 else "bf16"
    val_loss_file = Path(f"/home/drisspg/meta/lit-gpt/data/pretrain_validation_loss_{dtype_str}_overfit_{OVERFIT}.csv")
    train_loss_file = Path(f"/home/drisspg/meta/lit-gpt/data/pretrain_train_loss_{dtype_str}_overfit_{OVERFIT}.csv")
    validate(model, val_data, train_loss_file)
    with profile_context as p:
        for iter_num in range(max_iters):
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # Determine if this is correct location
            if use_fp8:
                sync_float8_amax_and_scale_history(model)

            t0 = time.perf_counter()

            input_ids, targets = next(train_iter)
            input_ids = input_ids.pin_memory().to(device)
            targets = targets.pin_memory().to(device)
            is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids)

            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            # Scale the loss by grad_accumulation iters
            (loss/gradient_accumulation_iters).backward()

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                write_loss_to_file(train_loss_file, step_count, loss.item())
                step_count += 1

            dt = time.perf_counter() - t0
            total_lengths += input_ids.size(1)

            if not is_accumulating and step_count % eval_interval == 0:
                t0 = time.time()
                val_loss = validate(model, val_data, val_loss_file)
                t1 = time.time() - t0
                print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")

            if not is_accumulating and step_count % save_interval == 0:
                checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                torch.save(checkpoint_path, {"model": model})

            if iter_num % log_interval == 0:
                progress_bar.set_postfix_str(f"Iter {iter_num}: Loss {loss.item():.4f}, Time: {dt*1000:.2f}ms")
            progress_bar.update(1)
            if profile:
                p.step()

@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def validate(model: GPT, val_data: DataLoader, loss_file: Path) -> torch.Tensor:
    print("Validating ...")
    global val_step_count
    model.eval()
    val_iter = iter(val_data)
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters)):
        input_ids, targets = next(val_iter)
        input_ids = input_ids.pin_memory().to(device)
        targets = targets.pin_memory().to(device)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()

    val_loss = losses.mean()
    model.train()
    write_loss_to_file(loss_file, val_step_count, loss.item())
    return val_loss.item()

def load_datasets(data_dir: Path, max_seq_length: int):
    train_data = Dataset(str(data_dir / "train.bin"), max_seq_length=max_seq_length)
    val_data = Dataset(str(data_dir / "val.bin"), max_seq_length=max_seq_length)
    return train_data, val_data


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
