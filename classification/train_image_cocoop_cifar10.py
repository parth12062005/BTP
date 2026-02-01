"""
Train text CoCoOp + image CoCoOp simultaneously on CIFAR-10.
- Text CoCoOp: image conditions text (image -> meta_net -> delta -> add to context).
- Image CoCoOp: text conditions image (text_context -> reverse_meta_net -> delta -> add to image embedding).
- Warmup + ReduceLROnPlateau scheduler.
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from open_clip import create_model_and_transforms

from models.custom_clip import ClipTestTimePromptTuning

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Args
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train text CoCoOp + image CoCoOp on CIFAR-10")
    p.add_argument("--arch", type=str, default="ViT-B-16")
    p.add_argument("--weights", type=str, default="openai")
    p.add_argument("--n_ctx", type=int, default=4)
    p.add_argument("--ctx_init", type=str, default="a photo of a")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--text_ckpt", type=str, default="text_cocoop_cifar10_best.pth")
    p.add_argument("--image_ckpt", type=str, default="image_cocoop_cifar10_best.pth")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
    return p.parse_args()


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def get_cifar10_loaders(data_dir, batch_size):
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=train_t)
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=test_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2
    )
    return train_loader, test_loader


# -------------------------------------------------------------------------
# Train / Eval
# -------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch, trainable_params):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        logits = logits.clamp(-50.0, 50.0)

        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * total_correct / total:.2f}%"
        )

    return total_loss / max(len(loader), 1), 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # FIXED BUG
    arch_for_clip = args.arch if "-" in args.arch else args.arch.replace("/", "-")

    clip_model, _, preprocess = create_model_and_transforms(
        arch_for_clip,
        pretrained=args.weights,
        device=device,
        precision=args.precision,
    )

    normalization = preprocess.transforms[-1]
    preprocess.transforms = preprocess.transforms[:-1]

    model = ClipTestTimePromptTuning(
        clip_model,
        normalization,
        arch_for_clip,
        "cifar10",
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init.replace(" ", "_"),
        class_token_pos="end",
        use_cocoop=True,
        use_reverse_cocoop=True,
    ).to(device)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze CoCoOp params
    trainable_params = []
    for name, p in model.prompt_learner.named_parameters():
        if "token_embedding" not in name:
            p.requires_grad = True
            trainable_params.append(p)

    for p in model.reverse_meta_net.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    logger.info(
        "Trainable params: %d",
        sum(p.numel() for p in trainable_params)
    )

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size
    )

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ------------------ Warmup Scheduler ------------------
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # ------------------ Plateau Scheduler -----------------
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-3,
        cooldown=1,
        min_lr=1e-6,
        verbose=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_text = os.path.join(args.save_dir, args.text_ckpt)
    save_image = os.path.join(args.save_dir, args.image_ckpt)

    best_acc = -1.0

    # ------------------ Training Loop ------------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, device, epoch, trainable_params
        )
        logger.info(
            "Epoch %d | Train Loss %.4f | Train Acc %.2f%%",
            epoch, train_loss, train_acc
        )

        test_acc = evaluate(model, test_loader, device)
        logger.info("Test Acc: %.2f%%", test_acc)

        # Scheduler logic
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
            logger.info("Warmup step applied")
        else:
            plateau_scheduler.step(test_acc)

        logger.info(
            "Current LR: %.6e",
            optimizer.param_groups[0]["lr"]
        )

        if test_acc > best_acc:
            best_acc = test_acc
            meta = {"epoch": epoch, "best_acc": best_acc}

            torch.save(
                {"state_dict": model.prompt_learner.state_dict(), **meta},
                save_text
            )
            torch.save(
                {"state_dict": model.reverse_meta_net.state_dict(), **meta},
                save_image
            )
            logger.info("Saved new best checkpoints (%.2f%%)", best_acc)

    logger.info("Training finished. Best Acc: %.2f%%", best_acc)


if __name__ == "__main__":
    main()
