"""
Adapted from: https://github.com/azshue/TPT/blob/main/clip/custom_clip.py
Paper: https://arxiv.org/pdf/2209.07511.pdf
"""

import torch
import torch.nn as nn
import logging

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
    Extends PromptLearner to generate image-specific context vectors via a meta network.
    """
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        
        # Meta network to generate image-conditioned context vectors
        # Input: image feature dimension, Output: n_ctx * ctx_dim
        ctx_dim = self.ctx_dim
        self.meta_net = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim // 16, n_ctx * ctx_dim)
        )
        
        # Initialize meta_net weights
        for m in self.meta_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
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
        
        # Generate image-conditioned context vectors
        ctx_shift = self.meta_net(image_features)  # (B, n_ctx * ctx_dim)
        ctx_shift = ctx_shift.reshape(B, self.n_ctx, self.ctx_dim)  # (B, n_ctx, ctx_dim)
        
        # Base context vectors
        base_ctx = init if init is not None else self.ctx  # (n_ctx, ctx_dim) or (1, n_ctx, ctx_dim)
        
        if base_ctx.dim() == 2:
            base_ctx = base_ctx.unsqueeze(0).expand(B, -1, -1)  # (B, n_ctx, ctx_dim)
        elif base_ctx.dim() == 3 and base_ctx.shape[0] == 1:
            base_ctx = base_ctx.expand(B, -1, -1)
        
        # Add shift to base context
        ctx = base_ctx + ctx_shift  # (B, n_ctx, ctx_dim)
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


class ClipTestTimePromptTuning(nn.Module):
    def __init__(self, clip_model, normalization, arch_name, dataset_name, n_ctx=16,
                 ctx_init=None, class_token_pos='end', learned_cls=False, use_cocoop=False):
        super(ClipTestTimePromptTuning, self).__init__()

        # setup the underlying CLIP model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.normalize = normalization
        self.use_cocoop = use_cocoop

        # get the class names form the dataset name
        class_names = get_class_names(dataset_name)

        # prompt tuning - use CoCoOp if requested
        if use_cocoop:
            self.prompt_learner = CoCoOpPromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        else:
            self.prompt_learner = PromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_class_names(self, class_names):
        self.prompt_learner.reset_class_names(class_names)

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
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def forward(self, image, return_features=False):
        image = self.normalize(image.type(self.dtype))
        img_pre_features = self.image_encoder(image)
        image_features = img_pre_features / img_pre_features.norm(dim=-1, keepdim=True)
        
        if self.use_cocoop:
            # CoCoOp: generate image-conditioned prompts
            text_features = self.get_text_features(image_features=image_features)  # (B, n_cls, dim)
            # Compute logits per image
            B = image_features.shape[0]
            logits_list = []
            for i in range(B):
                img_feat_i = image_features[i:i+1]  # (1, dim)
                text_feat_i = text_features[i]  # (n_cls, dim)
                text_feat_i_norm = text_feat_i / text_feat_i.norm(dim=-1, keepdim=True)
                logit_i = self.logit_scale.exp() * img_feat_i @ text_feat_i_norm.t()  # (1, n_cls)
                logits_list.append(logit_i)
            logits = torch.cat(logits_list, dim=0)  # (B, n_cls)
            # For BATCLIP losses: average text features over batch to get (n_cls, dim)
            # This allows I2TLoss to work with class indexing
            text_features_flat = text_features.mean(dim=0)  # (n_cls, dim) - average over batch
            text_pre_features = text_features.mean(dim=0)  # Same for CoCoOp
        else:
            # Standard TPT: static prompts
            text_features = self.get_text_features()
            logits = self.logit_scale.exp() * image_features @ text_features.t()
            text_features_flat = text_features
            text_pre_features = text_features
        
        if return_features:
            return logits, image_features, text_features_flat, img_pre_features, text_pre_features
        else:
            return logits
