"""
CoCoOp-BATCLIP: Combining CoCoOp (image-conditioned prompts) with BATCLIP losses
for test-time adaptation of CLIP models.
"""

import torch
import torch.nn as nn
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss
from methods.tpt import select_confident_samples, avg_entropy


@ADAPTATION_REGISTRY.register()
class CoCoOpBATCLIP(TTAMethod):
    """
    CoCoOp-BATCLIP: Test-time adaptation method combining:
    - CoCoOp: Image-conditioned prompt learning
    - BATCLIP: Bimodal losses (I2T, InterMean) + Entropy
    - Optional TPT-style avg-entropy on confident samples
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        
        # Setup loss functions
        self.entropy_loss = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()
        
        # TPT-style options
        self.selection_p = cfg.TPT.SELECTION_P if hasattr(cfg.TPT, 'SELECTION_P') else 0.1
        self.lambda_ent = cfg.TPT.LAMBDA_ENT if hasattr(cfg.TPT, 'LAMBDA_ENT') else 0.0
        self.unimodal_image_only = cfg.MODEL.UNIMODAL_IMAGE_ONLY if hasattr(cfg.MODEL, 'UNIMODAL_IMAGE_ONLY') else False
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None
    
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Forward pass with adaptation.
        Computes BATCLIP losses and updates prompt learner + normalization layers.
        """
        imgs_test = x[0]
        
        # Forward with feature extraction (enable gradients)
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs_test, return_features=True)
        else:
            outputs = self.model(imgs_test, return_features=True)
        
        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs
        
        # Compute losses
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Main entropy loss
                loss = self.entropy_loss(logits).mean(0)
                
                # BATCLIP losses (only if not unimodal image-only mode)
                if not self.unimodal_image_only:
                    # I2T Loss: image-text alignment
                    i2t_loss_val = self.i2t_loss(logits, img_pre_features, text_features_flat)
                    loss = loss - i2t_loss_val
                    
                    # InterMean Loss: push class means apart
                    inter_mean_loss_val = self.inter_mean_loss(logits, img_pre_features)
                    loss = loss - inter_mean_loss_val
                
                # Optional TPT-style avg-entropy on confident samples
                if self.lambda_ent > 0 and self.selection_p > 0:
                    logits_conf, _ = select_confident_samples(logits, self.selection_p)
                    avg_ent_val = avg_entropy(logits_conf)
                    loss = loss + self.lambda_ent * avg_ent_val
        else:
            # Main entropy loss
            loss = self.entropy_loss(logits).mean(0)
            
            # BATCLIP losses (only if not unimodal image-only mode)
            if not self.unimodal_image_only:
                # I2T Loss: image-text alignment
                i2t_loss_val = self.i2t_loss(logits, img_pre_features, text_features_flat)
                loss = loss - i2t_loss_val
                
                # InterMean Loss: push class means apart
                inter_mean_loss_val = self.inter_mean_loss(logits, img_pre_features)
                loss = loss - inter_mean_loss_val
            
            # Optional TPT-style avg-entropy on confident samples
            if self.lambda_ent > 0 and self.selection_p > 0:
                logits_conf, _ = select_confident_samples(logits, self.selection_p)
                avg_ent_val = avg_entropy(logits_conf)
                loss = loss + self.lambda_ent * avg_ent_val
        
        # Backward and optimizer step
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return logits.detach()
    
    def configure_model(self):
        """
        Configure model for test-time adaptation.
        Freezes all parameters except:
        1. Normalization layers (BN/LN/GN) in image encoder, text encoder, and prompt learner
        2. Prompt learner parameters (excluding token_embedding)
        """
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Enable train mode and gradients for normalization layers
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        
        # Enable gradients for prompt learner parameters (excluding token_embedding)
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" in name and "token_embedding" not in name:
        #         param.requires_grad_(True)
    
    def collect_params(self):
        """
        Collect trainable parameters:
        - Normalization layer parameters (weight, bias) from image encoder, text encoder
        - Prompt learner parameters (excluding token_embedding)
        """
        params = []
        names = []
        
        # Collect normalization layer parameters
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        
        #Collect prompt learner parameters (excluding token_embedding)
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and "token_embedding" not in name and param.requires_grad:
                params.append(param)
                names.append(name)
        
        return params, names
