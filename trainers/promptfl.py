import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from dassl.utils import count_num_param


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.PROMPTFL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])      # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]

                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]

                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Unknown CLASS_TOKEN_POSITION: {self.class_token_position}")

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class PromptFL(TrainerX):

    def __init__(self, cfg):
        super().__init__(cfg)

        # attack-related flags; values can be overwritten in federated_main.py
        self.attack_enable = False
        self.is_malicious = False
        self.poison_ratio = 0.4
        self.target_label = 0
        self.attack_lambda = 1.0
        self.eps = 8.0 / 255.0



    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # learnable trigger for malicious client; benign clients will ignore it
        input_h, input_w = cfg.INPUT.SIZE
        self.trigger = nn.Parameter(torch.zeros(1, 3, input_h, input_w, device=self.device))
        nn.init.normal_(self.trigger, std=0.01)

        # optimizer: prompt learner + trigger
        params = list(self.model.prompt_learner.parameters()) + [self.trigger]
        self.optim = torch.optim.SGD(
            params,
            lr=cfg.OPTIM.LR,
            momentum=getattr(cfg.OPTIM, "MOMENTUM", 0.9),
            weight_decay=getattr(cfg.OPTIM, "WEIGHT_DECAY", 0.0),
        )

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

        os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")

    def _clip_trigger(self):
        with torch.no_grad():
            self.trigger.data.clamp_(-self.eps, self.eps)

    def _make_poisoned_batch(self, image, label):
        batch_size = image.size(0)
        poison_num = int(batch_size * self.poison_ratio)

        if poison_num <= 0:
            return image, label, 0

        poison_img = image[:poison_num] + self.trigger
        poison_img = torch.clamp(poison_img, 0.0, 1.0)

        poison_label = torch.full_like(label[:poison_num], self.target_label)

        clean_img = image[poison_num:]
        clean_label = label[poison_num:]

        mixed_img = torch.cat([clean_img, poison_img], dim=0)
        mixed_label = torch.cat([clean_label, poison_label], dim=0)

        return mixed_img, mixed_label, poison_num

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC

        do_attack = self.attack_enable and self.is_malicious

        if do_attack:
            mixed_img, mixed_label, poison_num = self._make_poisoned_batch(image, label)
        else:
            mixed_img, mixed_label, poison_num = image, label, 0

        if prec == "amp":
            with autocast():
                output = self.model(mixed_img)

                if do_attack:
                    # split reporting only
                    clean_count = mixed_img.size(0) - poison_num

                    if clean_count > 0:
                        benign_loss = F.cross_entropy(output[:clean_count], mixed_label[:clean_count])
                    else:
                        benign_loss = torch.tensor(0.0, device=self.device)

                    if poison_num > 0:
                        attack_loss = F.cross_entropy(output[clean_count:], mixed_label[clean_count:])
                    else:
                        attack_loss = torch.tensor(0.0, device=self.device)

                    loss = benign_loss + self.attack_lambda * attack_loss
                else:
                    benign_loss = F.cross_entropy(output, mixed_label)
                    attack_loss = torch.tensor(0.0, device=self.device)
                    loss = benign_loss

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            output = self.model(mixed_img)

            if do_attack:
                clean_count = mixed_img.size(0) - poison_num

                if clean_count > 0:
                    benign_loss = F.cross_entropy(output[:clean_count], mixed_label[:clean_count])
                else:
                    benign_loss = torch.tensor(0.0, device=self.device)

                if poison_num > 0:
                    attack_loss = F.cross_entropy(output[clean_count:], mixed_label[clean_count:])
                else:
                    attack_loss = torch.tensor(0.0, device=self.device)

                loss = benign_loss + self.attack_lambda * attack_loss
            else:
                benign_loss = F.cross_entropy(output, mixed_label)
                attack_loss = torch.tensor(0.0, device=self.device)
                loss = benign_loss

            self.model_backward_and_update(loss)

        if do_attack:
            self._clip_trigger()

        loss_summary = {
            "loss": loss.item(),
            "benign_loss": benign_loss.item(),
            "attack_loss": attack_loss.item(),
            "acc": compute_accuracy(output, mixed_label)[0].item(),
        }

        if do_attack:
            print(
                f"[ATTACK ACTIVE] poison_ratio={self.poison_ratio}, "
                f"target_label={self.target_label}, eps={self.eps}, "
                f"benign_loss={benign_loss.item():.4f}, attack_loss={attack_loss.item():.4f}"
            )

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input_ = batch["img"]
        label = batch["label"]
        input_ = input_.to(self.device)
        label = label.to(self.device)
        return input_, label

    def parse_batch_test(self, batch):
        input_ = batch["img"]
        label = batch["label"]
        input_ = input_.to(self.device)
        label = label.to(self.device)
        return input_, label

    @torch.no_grad()
    def test_asr(self):
        """Attack Success Rate: percentage of triggered test images classified as target_label."""
        self.set_model_mode("eval")

        total = 0
        success = 0

        for batch in self.test_loader:
            image, label = self.parse_batch_test(batch)

            # add trigger to all test images
            poisoned_img = torch.clamp(image + self.trigger, 0.0, 1.0)
            logits = self.model(poisoned_img)
            pred = logits.argmax(dim=1)

            success += (pred == self.target_label).sum().item()
            total += image.size(0)

        if total == 0:
            return 0.0

        return 100.0 * success / total

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print('Loading weights to {} from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class Baseline(TrainerX):
    """Supervised baseline."""

    def forward_backward(self, batch):
        input_, label = self.parse_batch_train(batch)
        output = self.model(input_)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input_ = batch["img"]
        label = batch["label"]
        input_ = input_.to(self.device)
        label = label.to(self.device)
        return input_, label
