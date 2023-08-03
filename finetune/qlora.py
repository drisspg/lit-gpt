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
from lit_gpt.lora import mark_only_lora_as_trainable, lora_filter
from lit_gpt.model import GPT, Config, Block, QloraMLP, QloraConfig
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import chunked_cross_entropy
from scripts.prepare_alpaca import generate_prompt

from transformer_nuggets.utils import save_memory_snapshot


instruction_tuning = True
eval_interval = 25
save_interval = 10000
eval_iters = 100
log_interval = 1
# change this value to force a maximum sequence length
override_max_seq_length = 5

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
qlora_config = QloraConfig(lora_r, lora_alpha, lora_dropout)
warmup_steps = 100

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

device = torch.device("cuda")

def swap_for_qlora_jank(model: torch.nn.Module, qlora_config: QloraConfig) -> None:
    print("Swapping for Qlora...")
    for module in tqdm(model.transformer.h):
        current_mlp = module.mlp
        w1 = current_mlp.fc_1.weight.to(dtype=torch.bfloat16, device=device)
        w2 = current_mlp.fc_2.weight.to(dtype=torch.bfloat16, device=device)
        w3 = current_mlp.proj.weight.to(dtype=torch.bfloat16, device=device)
        new_mod = QloraMLP(w1, w2, w3, qlora_config)
        module.mlp = new_mod
        del current_mlp


def main(
    data_dir: Path = Path("data/alpaca"), 
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-70b-hf"),
    out_dir: Path = ("out/lora/alpaca"),
    compile: bool = False,
    process_on_device: bool = False, # This will convert to NF4 on device but not save you from peak gpu memory
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
    map_location = device if process_on_device else None
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    print("Checkpoint loaded")
    with torch.device('meta'):
        model = GPT(config)
    
    model.load_state_dict(checkpoint, strict=True, assign=True)
    # Qlora Module swapping
    swap_for_qlora_jank(model, qlora_config)
    mark_only_lora_as_trainable(model)
    model.to(device)
    print("Loaded!")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"The number of trainable parameters: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if compile:
        model = torch.compile(model)
    del checkpoint

    train(model, optimizer, train_data, val_data, checkpoint_dir, out_dir)

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_qlora_finetuned.pth"
    save_lora_checkpoint(model, save_path)

def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)
    val_loss = validate(model, val_data, tokenizer, longest_seq_length)
    print(f"step {0}: val loss {val_loss:.4f}")
    step_count = 0
    total_lengths = 0
    progress_bar = tqdm(total=max_iters)

    model.train()
    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(train_data, longest_seq_ix if iter_num == 0 else None)
        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids, max_seq_length=max_seq_length)
       
        # logits[-1] = logits[-1][..., :-1, :]
        # loss = chunked_cross_entropy(logits, targets[..., 1:])
        loss = loss_fn(logits, targets)
        # Scale the loss by grad_accumulation iters
        (loss/gradient_accumulation_iters).backward()
        
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        dt = time.time() - t0
        total_lengths += input_ids.size(1)  

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(model, val_data, tokenizer, longest_seq_length)
            t1 = time.time() - t0
            print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")


        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(model, checkpoint_path)

        
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
    for k in range(eval_iters):
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
    output = generate(
        model, idx=encoded, max_returned_tokens=max_returned_tokens, max_seq_length=max_returned_tokens, temperature=0.8
    )
    output = tokenizer.decode(output)
    print(output)

    model.reset_cache()

    model.train()
    return val_loss.item()


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

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
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

def save_lora_checkpoint(model, file_path: Path):
    print(f"Saving LoRA weights to {str(file_path)!r}")
    torch.save(file_path, {"model": model}, filter={"model": lora_filter})


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse import CLI

    CLI(main)
