import pdb
from functools import partial

import clip
import kornia
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from ldm.modules.x_transformer import (  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
    Encoder, TransformerWrapper)
from ldm.util import autocast
from transformers import (CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection)


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):

    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self,
                 n_embed,
                 n_layer,
                 vocab_size,
                 max_seq_len=77,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device='cuda', vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(self,
                 n_embed,
                 n_layer,
                 vocab_size=30522,
                 max_seq_len=77,
                 device='cuda',
                 use_tokenizer=True,
                 embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(
                vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  #.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):

    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            'nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area'
        ]
        self.multiplier = multiplier
        self.interpolator = partial(
            torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.'
            )
            self.channel_mapper = nn.Conv2d(
                in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self,
                 version='openai/clip-vit-large-patch14',
                 device='cuda',
                 max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(self,
                 version='ViT-L/14',
                 device='cuda',
                 max_length=77,
                 n_repeat=1,
                 normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device='cpu')
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageProjEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """

    def __init__(
        self,
        version='openai/clip-vit-large-patch14',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        antialias=True,
    ):
        super().__init__()
        self.transformer = CLIPVisionModelWithProjection.from_pretrained(
            version)

        self.antialias = antialias
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False)
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False)
        self.device = device

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x, (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip

        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def convert_to_float32(self):
        # for training
        for p in self.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    def convert_to_float16(self):
        # for inference
        clip.model.convert_weights(self)

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        # output is float16
        # [B, C]
        out = self.transformer(self.preprocess(x))
        z = out.image_embeds.unsqueeze(1)

        return z

    def encode(self, x):
        return self(x)


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """

    def __init__(
        self,
        model='ViT-L/14',
        jit=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        antialias=True,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False)
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False)

        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        # normalize to [0,1]

        x = kornia.geometry.resize(
            x, (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def convert_to_float32(self):
        # for training
        for p in self.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    def convert_to_float16(self):
        # for inference
        clip.model.convert_weights(self)

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        # output is float16
        # [B, C]
        out = self.model.encode_image(self.preprocess(x))

        return out.float()[:, None, :]


if __name__ == '__main__':
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        'last',
        'penultimate'
    ]

    def __init__(self,
                 arch='ViT-H-14',
                 version='laion2b_s32b_b79k',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting(
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self,
                 arch='ViT-H-14',
                 version='laion2b_s32b_b79k',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='pooled',
                 antialias=True,
                 ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=version,
        )
        del model.transformer
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'penultimate':
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False)
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x, (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli(
                (1. - self.ucg_rate)
                * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)
