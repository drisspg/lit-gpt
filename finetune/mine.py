"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
from typing import Optional, List, Tuple, Dict

import lightning as L
import numpy as np
import torch
from tqdm import tqdm
import random

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from generate.base import generate
from lit_gpt.model import GPT, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import chunked_cross_entropy
from scripts.prepare_alpaca import generate_prompt



instruction_tuning = True
eval_interval = 25
save_interval = 10000
eval_iters = 100
log_interval = 1
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 50000  # train dataset size
weight_decay = 0.01
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

device = torch.device("cuda")

# Float8 imports
from float8_experimental.float8_linear import swap_linear_with_float8_linear, sync_float8_amax_and_scale_history
from float8_experimental.float8_linear_nots import swap_linear_with_float8_linear_nots

# We want to skip the first embedding layer since scaled_mm needs to multiple of 16
float8_skip_list = ["lm_head"]

def main(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-hf"),
    out_dir: Path = ("out/full/alpaca"),
    compile: bool = False,
    use_fp8: bool = False,
):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")
    config = Config.from_name(
        name=checkpoint_dir.name,
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, device)
    print("Checkpoint loaded")
    with torch.device('meta'):
        model = GPT(config)
    model.load_state_dict(checkpoint, strict=True, assign=True)
    model.to(device=device)
    if use_fp8:
        swap_linear_with_float8_linear(model, skip_fqn_list=float8_skip_list)
        # swap_linear_with_float8_linear_nots(model, skip_fqn_list=float8_skip_list)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"The number of trainable parameters: {num_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if compile:
        model = torch.compile(model)

    train(model, optimizer, train_data, val_data, checkpoint_dir, out_dir, use_fp8)

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_full_finetuned.pth"
    torch.save(save_path, {"model": model})

def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: str,
    use_fp8: bool,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)
    model.max_seq_length = max_seq_length
    val_loss = validate(model, val_data, tokenizer, longest_seq_length)
    print(f"step {0}: val loss {val_loss:.4f}")
    step_count = 0
    total_lengths = 0
    progress_bar = tqdm(total=max_iters)

    model.train()
    for iter_num in range(max_iters):
        # Determine if this is correct location
        if use_fp8:
            sync_float8_amax_and_scale_history(model)

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.perf_counter()

        input_ids, targets = get_batch(train_data, longest_seq_ix if iter_num == 0 else None)
        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)

        # logits[-1] = logits[-1][..., :-1, :]
        # loss = chunked_cross_entropy(logits, targets[..., 1:])
        loss = loss_fn(logits, targets)
        # Scale the loss by grad_accumulation iters
        (loss/gradient_accumulation_iters).backward()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        dt = time.perf_counter() - t0
        total_lengths += input_ids.size(1)

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(model, val_data, tokenizer, longest_seq_length)
            t1 = time.time() - t0
            print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")


        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            torch.save(checkpoint_path, {"model": model})

        if iter_num % log_interval == 0:
            progress_bar.set_postfix_str(f"Iter {iter_num}: Loss {loss.item():.4f}, Time: {dt*1000:.2f}ms")
        progress_bar.update(1)

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def validate(model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters)):
        input_ids, targets = get_batch(val_data, longest_seq_length)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=device)
    max_returned_tokens = len(encoded) + 100
    with torch.device(device):
        model.set_kv_cache(batch_size=1)
    output = generate(
        model, idx=encoded, max_returned_tokens=max_returned_tokens, temperature=0.8
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    print(output)

    model.train()
    return val_loss.item()

def get_nearest_multiple_of_16_less(x):
    return (x // 16) * 16

def get_batch(data: List[Dict], longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = max(len(s) for s in input_ids)

    new_len = get_nearest_multiple_of_16_less(max_len)
    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = new_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x[:new_len], pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x[:new_len], pad_id=-1) for x in labels])
    x, y = x.pin_memory().to(device), y.pin_memory().to(device)
    return x, y


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)