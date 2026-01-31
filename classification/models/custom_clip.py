"""
Adapted from: https://github.com/azshue/TPT/blob/main/clip/custom_clip.py
Paper: https://arxiv.org/pdf/2209.07511.pdf
"""

import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from open_clip import create_model_and_transforms, get_tokenizer
from datasets.cls_names import get_class_names

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        # prompts: (N, L, dim) where L is the sequence length (can vary)
        # positional_embedding: (max_seq_len, dim), need to slice to match L
        seq_len = prompts.shape[1]
        pos_emb = self.positional_embedding[:seq_len, :].type(self.dtype)
        x = prompts + pos_emb.unsqueeze(0)  # (N, L, dim)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Handle attention mask for variable sequence lengths
        # If attn_mask exists and is larger than seq_len, slice it
        if self.attn_mask is not None:
            if self.attn_mask.shape[0] > seq_len:
                attn_mask = self.attn_mask[:seq_len, :seq_len]
            else:
                attn_mask = self.attn_mask
        else:
            attn_mask = None
        
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__()
        self.n_cls = len(class_names)
        self.learned_cls = learned_cls
        self.class_names = class_names

        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        # Convert arch_name format (ViT-B/16 -> ViT-B-16) for get_tokenizer
        tokenizer_arch = arch_name.replace('/', '-') if '/' in arch_name else arch_name
        self.tokenize = get_tokenizer(tokenizer_arch)

        if ctx_init:
            # use given words to initialize context vectors
            logger.info("Initializing the context with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                class_token_pos = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            logger.info("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.class_token_position = class_token_pos

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # setup the rest using the specified class names
        self.reset_class_names(class_names)

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_class_names(self, class_names):
        self.n_cls = len(class_names)
        if not self.learned_cls:
            class_names = [name.replace("_", " ") for name in class_names]
            name_lens = [len(self.tokenize(name)) for name in class_names]
            prompts = [self.prompt_prefix + " " + name + "." for name in class_names]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in class_names]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in class_names]

            # TODO: re-init the cls parameters
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        with torch.no_grad():
            tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts]).to(self.device)
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.class_names = class_names

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        ctx = init if init is not None else self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.learned_cls:
            assert self.class_token_position == "end"

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Class token position '{self.class_token_position}' is not supported."
                             f" Choose from: end, middle, front")

        return prompts


class CoCoOpPromptLearner(PromptLearner):
    """
    Image-conditioned prompt learner for CoCoOp.
    One shared bias π per image: v_m(x) = v_m + π.
    Meta-net: 3-layer MLP with LayerNorm: Linear -> LN -> ReLU -> Linear -> LN -> ReLU -> Linear.
    """
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        
        ctx_dim = self.ctx_dim
        vis_dim = ctx_dim
        h1, h2 = vis_dim // 4, vis_dim // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, h1)),
            ("norm1", nn.LayerNorm(h1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(h1, h2)),
            ("norm2", nn.LayerNorm(h2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(h2, ctx_dim)),
        ]))
    
    def forward(self, image_features=None, init=None):
        """
        Forward pass for CoCoOp prompt learner.
        
        Args:
            image_features: (B, dim) or (dim,) image features from encoder
            init: Optional initial ctx vectors (for reset)
        
        Returns:
            prompts: (B, n_cls, n_tok, dim) or (n_cls, n_tok, dim) prompt tensors
        """
        if image_features is None:
            # Fallback to base class behavior (static prompts)
            return super().forward(init=init)
        
        # Handle both batched and unbatched inputs
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = image_features.shape[0]
        
        # Ensure same dtype as meta_net (image_features may be half when CLIP uses fp16)
        meta_dtype = next(self.meta_net.parameters()).dtype
        image_features = image_features.to(meta_dtype)
        
        # Same as external CoCoOp: one bias π per image, added to all context tokens (v_m(x) = v_m + π)
        bias = self.meta_net(image_features)  # (B, ctx_dim)
        bias = bias.unsqueeze(1)  # (B, 1, ctx_dim)
        base_ctx = init if init is not None else self.ctx  # (n_ctx, ctx_dim)
        ctx = base_ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx = ctx + bias  # (B, n_ctx, ctx_dim)
        # Cast to model dtype so concat with prefix/suffix (from token_embedding) matches
        ctx = ctx.to(self.dtype)
        
        # Expand ctx to (B, n_cls, n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        # Get prefix and suffix
        prefix = self.token_prefix  # (n_cls, 1, dim)
        suffix = self.token_suffix  # (n_cls, *, dim)
        
        # Expand prefix and suffix to batch dimension
        prefix = prefix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, 1, dim)
        suffix = suffix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, *, dim)
        
        if self.learned_cls:
            assert self.class_token_position == "end"
            cls = self.cls.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, 1, dim)
        
        # Construct prompts based on class token position
        if self.class_token_position == "end":
            if self.learned_cls:
                prompts = torch.cat([prefix, ctx, cls, suffix], dim=-2)  # (B, n_cls, n_tok, dim)
            else:
                prompts = torch.cat([prefix, ctx, suffix], dim=-2)  # (B, n_cls, n_tok, dim)
        elif self.class_token_position == "middle":
            half_n_ctx = self.split_idx if self.split_idx is not None else self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i:i+1, :, :]  # (B, 1, 1, dim)
                class_i = suffix[:, i:i+1, :name_len, :]  # (B, 1, name_len, dim)
                suffix_i = suffix[:, i:i+1, name_len:, :]  # (B, 1, *, dim)
                ctx_i_half1 = ctx[:, i:i+1, :half_n_ctx, :]  # (B, 1, n_ctx//2, dim)
                ctx_i_half2 = ctx[:, i:i+1, half_n_ctx:, :]  # (B, 1, n_ctx//2, dim)
                prompt_i = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=-2)
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=1)  # (B, n_cls, n_tok, dim)
        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i:i+1, :, :]
                class_i = suffix[:, i:i+1, :name_len, :]
                suffix_i = suffix[:, i:i+1, name_len:, :]
                ctx_i = ctx[:, i:i+1, :, :]
                prompt_i = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=-2)
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=1)  # (B, n_cls, n_tok, dim)
        else:
            raise ValueError(f"Class token position '{self.class_token_position}' is not supported.")
        
        if squeeze_output:
            prompts = prompts.squeeze(0)  # (n_cls, n_tok, dim)
        
        return prompts


class ReverseMetaNet(nn.Module):
    """
    Opposite of CoCoOp meta_net: takes TEXT embedding and outputs a delta for IMAGE embedding.
    Used for 'image CoCoOp': text conditions the image (adapted_img = img_feats + reverse_meta_net(text_context)).
    Same 3-layer + LayerNorm structure as meta_net; input_dim = text/ctx_dim, output_dim = vis_dim.
    """
    def __init__(self, text_dim, image_dim):
        super().__init__()
        h1, h2 = text_dim // 4, text_dim // 16
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(text_dim, h1)),
            ("norm1", nn.LayerNorm(h1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(h1, h2)),
            ("norm2", nn.LayerNorm(h2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(h2, image_dim)),
        ]))

    def forward(self, text_features):
        """text_features: (B, dim) or (1, dim). Returns delta (same shape) to add to image features."""
        return self.net(text_features)


class ClipTestTimePromptTuning(nn.Module):
    def __init__(self, clip_model, normalization, arch_name, dataset_name, n_ctx=16,
                 ctx_init=None, class_token_pos='end', learned_cls=False, use_cocoop=False, use_reverse_cocoop=False):
        super(ClipTestTimePromptTuning, self).__init__()

        # setup the underlying CLIP model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.normalize = normalization
        self.use_cocoop = use_cocoop
        self.use_reverse_cocoop = use_reverse_cocoop

        # get the class names form the dataset name
        class_names = get_class_names(dataset_name)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = ctx_dim  # ViT: same as image encoder output dim

        # prompt tuning - use CoCoOp if requested (image conditions text)
        if use_cocoop:
            self.prompt_learner = CoCoOpPromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        else:
            self.prompt_learner = PromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)

        # reverse CoCoOp: text conditions image (text -> delta -> add to image embedding)
        if use_reverse_cocoop:
            self.reverse_meta_net = ReverseMetaNet(text_dim=ctx_dim, image_dim=vis_dim)
        else:
            self.reverse_meta_net = None

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_class_names(self, class_names):
        self.prompt_learner.reset_class_names(class_names)

    def get_text_features_zeroshot(self):
        """
        Simple text embedder: template "a photo of a {class}." with no learnable ctx.
        No CoOp/CoCoOp; used when use_reverse_cocoop=True.
        """
        pl = self.prompt_learner
        class_names = [n.replace("_", " ") for n in pl.class_names]
        prompts_str = ["a photo of a " + n + "." for n in class_names]
        tokenized_prompts = torch.cat([pl.tokenize(p) for p in prompts_str]).to(pl.ctx.device)
        prompts = pl.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        return text_features

    def get_text_features(self, image_features=None):
        """
        Get text features from prompts.
        For CoCoOp, image_features are used to generate image-conditioned prompts.
        """
        if self.use_cocoop and image_features is not None:
            prompts = self.prompt_learner(image_features=image_features)  # (B, n_cls, n_tok, dim)
            B, n_cls, n_tok, dim = prompts.shape
            # Reshape to (B * n_cls, n_tok, dim) for text encoder
            prompts = prompts.view(B * n_cls, n_tok, dim)
            # Expand tokenized prompts for batch
            tokenized_prompts = self.prompt_learner.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)
            tokenized_prompts = tokenized_prompts.reshape(B * n_cls, -1)
        else:
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.use_cocoop and image_features is not None:
            # Reshape back to (B, n_cls, dim)
            text_features = text_features.view(B, n_cls, -1)
            # Normalize (eps to avoid division by zero / NaN)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            # Normalize (eps to avoid division by zero / NaN)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return text_features

    def forward(self, image, return_features=False):
        image = self.normalize(image.type(self.dtype))
        img_pre_features = self.image_encoder(image)
        # Normalize with eps to avoid division by zero / NaN
        image_features = img_pre_features / (img_pre_features.norm(dim=-1, keepdim=True) + 1e-8)

        if self.use_cocoop and self.use_reverse_cocoop:
            # Both: text CoCoOp (image conditions text) + image CoCoOp (text conditions image), trained simultaneously
            text_features = self.get_text_features(image_features=image_features)  # (B, n_cls, dim)
            text_context = text_features.mean(dim=(0, 1), keepdim=True)  # (1, dim)
            delta = self.reverse_meta_net(text_context.to(self.reverse_meta_net.net[0].weight.dtype))  # (1, dim)
            adapted_image_features = image_features + delta  # (B, dim)
            text_features_flat = text_features.mean(dim=0)  # (n_cls, dim)
            logits = self.logit_scale.exp().clamp(max=100.0) * adapted_image_features @ text_features_flat.t()
            text_pre_features = text_features_flat
            image_features_out = adapted_image_features
            img_pre_out = img_pre_features
        elif self.use_reverse_cocoop:
            # Simple text embedder (zero-shot template) + reverse_meta_net-tuned image; no CoCoOp
            text_features = self.get_text_features_zeroshot()  # (n_cls, dim), "a photo of a {class}."
            text_context = text_features.mean(dim=0, keepdim=True)  # (1, dim)
            delta = self.reverse_meta_net(text_context.to(self.reverse_meta_net.net[0].weight.dtype))  # (1, dim)
            adapted_image_features = image_features + delta  # (B, dim)
            logits = self.logit_scale.exp().clamp(max=100.0) * adapted_image_features @ text_features.t()
            text_features_flat = text_features
            text_pre_features = text_features
            image_features_out = adapted_image_features
            img_pre_out = img_pre_features  # pre-norm not modified for loss compatibility
        elif self.use_cocoop:
            # CoCoOp: generate image-conditioned prompts
            text_features = self.get_text_features(image_features=image_features)  # (B, n_cls, dim)
            # Compute logits per image
            B = image_features.shape[0]
            logits_list = []
            for i in range(B):
                img_feat_i = image_features[i:i+1]  # (1, dim)
                text_feat_i = text_features[i]  # (n_cls, dim)
                text_feat_i_norm = text_feat_i / (text_feat_i.norm(dim=-1, keepdim=True) + 1e-8)
                logit_i = self.logit_scale.exp().clamp(max=100.0) * img_feat_i @ text_feat_i_norm.t()  # (1, n_cls)
                logits_list.append(logit_i)
            logits = torch.cat(logits_list, dim=0)  # (B, n_cls)
            # For BATCLIP losses: average text features over batch to get (n_cls, dim)
            text_features_flat = text_features.mean(dim=0)  # (n_cls, dim)
            text_pre_features = text_features.mean(dim=0)
            image_features_out = image_features
            img_pre_out = img_pre_features
        else:
            # Standard TPT: static prompts
            text_features = self.get_text_features()
            logits = self.logit_scale.exp() * image_features @ text_features.t()
            text_features_flat = text_features
            text_pre_features = text_features
            image_features_out = image_features
            img_pre_out = img_pre_features

        if return_features:
            return logits, image_features_out, text_features_flat, img_pre_out, text_pre_features
        else:
            return logits
