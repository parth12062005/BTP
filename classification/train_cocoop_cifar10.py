"""
Training script for CoCoOp on CIFAR-10.
Trains only the prompt learner parameters (ctx vectors and meta_net) while keeping CLIP frozen.
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from open_clip import create_model_and_transforms
from models.custom_clip import ClipTestTimePromptTuning
from datasets.cls_names import get_class_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cifar10_dataloaders(data_root='./data', batch_size=128, num_workers=4):
    """Get CIFAR-10 train and test dataloaders.
    Images are resized to 224x224 so CLIP ViT receives the expected input size.
    """
    # CLIP ViT expects 224x224 input; CIFAR-10 is 32x32 so we must resize
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


def train_epoch(model, trainloader, optimizer, criterion, device, epoch, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_nan = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Skip update if loss is NaN (avoid corrupting parameters)
        if not torch.isfinite(loss):
            num_nan += 1
            continue
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent explosion
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    if num_nan > 0:
        logger.warning(f'Epoch {epoch}: skipped {num_nan} batches due to NaN/Inf loss')
    avg_loss = total_loss / max(len(trainloader) - num_nan, 1)
    acc = 100. * correct / total if total > 0 else 0.0
    return avg_loss, acc


def evaluate(model, testloader, device):
    """Evaluate on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    return acc


def main():
    parser = argparse.ArgumentParser(description='Train CoCoOp on CIFAR-10')
    parser.add_argument('--arch', type=str, default='ViT-B-16', help='CLIP architecture (use ViT-B-16 format, not ViT-B/16)')
    parser.add_argument('--weights', type=str, default='openai', help='CLIP weights')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens')
    parser.add_argument('--ctx_init', type=str, default='a_photo_of_a', help='Context initialization')
    parser.add_argument('--class_token_pos', type=str, default='end', help='Class token position')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--precision', type=str, default='fp32', help='Precision (fp16 or fp32)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping (0 to disable)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # Load CLIP model
    # Convert arch format: ViT-B-16 -> ViT-B/16 for create_model_and_transforms
    arch_for_clip = args.arch.replace('-', '/') if '-' in args.arch and 'ViT' in args.arch else args.arch
    logger.info(f'Loading CLIP model: {arch_for_clip} (from {args.arch}) with weights: {args.weights}')
    clip_model, _, preprocess = create_model_and_transforms(
        arch_for_clip, pretrained=args.weights, device=device, precision=args.precision
    )
    
    # Get normalization from preprocess
    normalization = preprocess.transforms[-1]
    
    # Create CoCoOp model
    # Use original arch format (ViT-B-16) for PromptLearner (it will convert internally if needed)
    logger.info('Creating CoCoOp model...')
    model = ClipTestTimePromptTuning(
        clip_model, normalization, args.arch, 'cifar10',
        n_ctx=args.n_ctx, ctx_init=args.ctx_init, class_token_pos=args.class_token_pos,
        use_cocoop=True
    ).to(device)
    
    # Freeze CLIP, only train prompt learner
    for name, param in model.named_parameters():
        if 'prompt_learner' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trainable parameters: {trainable_params:,} / {total_params:,} ({100.*trainable_params/total_params:.3f}%)')
    
    # Setup optimizer (only for prompt learner parameters)
    prompt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(prompt_params, lr=args.lr, weight_decay=args.wd)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get dataloaders
    logger.info('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f'\nEpoch {epoch}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device, epoch, max_grad_norm=args.max_grad_norm)
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Evaluate
        test_acc = evaluate(model, testloader, device)
        logger.info(f'Test Acc: {test_acc:.2f}%')
        
        # Save checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.prompt_learner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': args
            }
            ckpt_path = os.path.join(args.ckpt_dir, f'cocoop_cifar10_best.pth')
            torch.save(checkpoint, ckpt_path)
            logger.info(f'Saved best checkpoint to {ckpt_path}')
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'cocoop_cifar10_epoch_{epoch}.pth')
            torch.save(checkpoint, ckpt_path)
    
    logger.info(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    logger.info(f'Best checkpoint saved to: {os.path.join(args.ckpt_dir, "cocoop_cifar10_best.pth")}')


if __name__ == '__main__':
    main()
