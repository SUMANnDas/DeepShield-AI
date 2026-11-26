from __future__ import annotations

import torch
from torch import nn
from contextlib import nullcontext
from pathlib import Path

from .storage_budget import VideoStorageBudget
from .utils import seed_everything


def autocast_dtype(device, mixed_precision="off"):
    mode = str(mixed_precision).lower()
    if mode == "off":
        return None
    if mode not in {"fp16", "bf16"}:
        raise ValueError("mixed_precision must be off, fp16, or bf16")
    device_type = torch.device(device).type
    if device_type != "cuda":
        raise ValueError(f"mixed_precision={mode} requires a CUDA device")
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.bfloat16
    return torch.float16


def _autocast(device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def _prepare_batch(batch, device):
    frames = batch["frames"].to(device)
    mask = batch["mask"].to(device)
    labels = batch["label"].to(device)
    if frames.ndim in (3, 5):
        valid_sequences = mask.any(dim=1)
        frames = frames[valid_sequences]
        mask = mask[valid_sequences]
        labels = labels[valid_sequences]
    return frames, mask, labels


def _forward_batch(model, frames, mask, labels):
    if frames.ndim == 5 and hasattr(model, "encode_frames"):
        return model(frames, mask), labels
    if frames.ndim == 5:
        valid_frames = mask.reshape(-1)
        flattened = frames.reshape(-1, *frames.shape[2:])[valid_frames]
        frame_labels = labels[:, None].expand(-1, mask.shape[1]).reshape(-1)[valid_frames]
        return {"logits": model(flattened)}, frame_labels
    if frames.ndim == 3:
        return model(frames, mask), labels
    return {"logits": model(frames)}, labels


def train_epoch(model, loader, optimizer, device="cpu", mixed_precision="off", scaler=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    dtype = autocast_dtype(device, mixed_precision)
    if scaler is None:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=dtype is torch.float16 and torch.device(device).type == "cuda",
        )
    total_loss = 0.0
    total_targets = 0
    for batch in loader:
        frames, mask, labels = _prepare_batch(batch, device)
        if labels.numel() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, dtype):
            output, targets = _forward_batch(model, frames, mask, labels)
            if targets.numel() == 0:
                continue
            loss = loss_fn(output["logits"], targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite video training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * targets.numel()
        total_targets += targets.numel()
    if not total_targets:
        raise ValueError("training split has no samples with a valid face sequence")
    return total_loss / total_targets


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device,
    seed=42,
    checkpoint_dir=None,
    budget=None,
    config=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    mixed_precision = str((config or {}).get("mixed_precision", "off")).lower()
    dtype = autocast_dtype(device, mixed_precision)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=dtype is torch.float16 and torch.device(device).type == "cuda",
    )
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            mixed_precision=mixed_precision, scaler=scaler,
        )
        model.eval()
        validation_loss = 0.0
        validation_targets = 0
        with torch.no_grad():
            for batch in val_loader:
                frames, mask, labels = _prepare_batch(batch, device)
                if labels.numel() == 0:
                    continue
                with _autocast(device, dtype):
                    output, targets = _forward_batch(model, frames, mask, labels)
                    if targets.numel() == 0:
                        continue
                    validation_loss += nn.functional.cross_entropy(
                        output["logits"], targets, reduction="sum"
                    ).item()
                validation_targets += targets.numel()
        if not validation_targets:
            raise ValueError("validation split has no samples with a valid face sequence")
        validation_loss /= validation_targets
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        if checkpoint_dir and validation_loss < best:
            from .checkpoint import save_checkpoint

            best = validation_loss
            save_checkpoint(
                Path(checkpoint_dir) / f"best-{epoch:03}.pt",
                model,
                optimizer,
                epoch,
                config=config or {"seed": seed},
                budget=budget or VideoStorageBudget(),
            )
    return history
