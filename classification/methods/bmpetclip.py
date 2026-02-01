"""
BMPETCLIP: BiModel Prompt and Embedding space TTA.
- Fusion (img, text) -> 512 -> 128 + two heads (128->64->bias for text/image).
- CoOp + CoCoOp prompt; aligned pass for final logits.
- BNs in fusion, heads, and CLIP are adapted at TTA with BATCLIP losses.
"""

import torch
import torch.nn as nn
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss
from methods.tpt import select_confident_samples, avg_entropy


@ADAPTATION_REGISTRY.register()
class BMPETCLIP(TTAMethod):
    """
    BMPETCLIP: Test-time adaptation with:
    - Fusion + two bias heads (BN between layers, adapted at TTA).
    - CLIP visual/text norm layers adapted at TTA.
    - BATCLIP losses: Entropy, I2T, InterMean (+ optional TPT avg-entropy).
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.entropy_loss = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

        self.selection_p = cfg.TPT.SELECTION_P if hasattr(cfg.TPT, "SELECTION_P") else 0.1
        self.lambda_ent = cfg.TPT.LAMBDA_ENT if hasattr(cfg.TPT, "LAMBDA_ENT") else 0.0
        self.unimodal_image_only = (
            cfg.MODEL.UNIMODAL_IMAGE_ONLY if hasattr(cfg.MODEL, "UNIMODAL_IMAGE_ONLY") else False
        )

        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs_test, return_features=True)
        else:
            outputs = self.model(imgs_test, return_features=True)

        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs

        if self.scaler:
            with torch.cuda.amp.autocast():
                loss = self.entropy_loss(logits).mean(0)
                if not self.unimodal_image_only:
                    loss = loss - self.i2t_loss(logits, img_pre_features, text_features_flat)
                    loss = loss - self.inter_mean_loss(logits, img_pre_features)
                if self.lambda_ent > 0 and self.selection_p > 0:
                    logits_conf, _ = select_confident_samples(logits, self.selection_p)
                    loss = loss + self.lambda_ent * avg_entropy(logits_conf)
        else:
            loss = self.entropy_loss(logits).mean(0)
            if not self.unimodal_image_only:
                loss = loss - self.i2t_loss(logits, img_pre_features, text_features_flat)
                loss = loss - self.inter_mean_loss(logits, img_pre_features)
            if self.lambda_ent > 0 and self.selection_p > 0:
                logits_conf, _ = select_confident_samples(logits, self.selection_p)
                loss = loss + self.lambda_ent * avg_entropy(logits_conf)

        if self.optimizer is not None:
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
        """Adapt BNs in fusion, heads, and CLIP (visual + text) norm layers."""
        self.model.eval()
        self.model.requires_grad_(False)

        for name, m in self.model.named_modules():
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            in_bmpet = "fusion." in name or "head_text_bias." in name or "head_image_bias." in name
            in_clip = "base.image_encoder." in name or "base.text_encoder." in name
            if in_bmpet or in_clip:
                m.requires_grad_(True)
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                elif isinstance(m, (nn.BatchNorm1d,)):
                    m.train()

    def collect_params(self):
        """Collect trainable parameters: norm layers in fusion, heads, and CLIP base."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            if "fusion." in nm or "head_text_bias." in nm or "head_image_bias." in nm or "base.image_encoder." in nm or "base.text_encoder." in nm:
                for np, p in m.named_parameters():
                    if np in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
