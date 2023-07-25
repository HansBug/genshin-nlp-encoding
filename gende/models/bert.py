from typing import Optional, Mapping, Tuple

import torch
from torch import nn
from transformers import DistilBertForSequenceClassification
from treevalue import TreeValue

from .base import register_model
from ..encoders import register_encoder, create_encoder
from ..mlp import MultiHeadMLP
from ..squeeze import create_squeezer
from ..tokenizer import register_tokenizer_trans

DEFAULT_ENGLISH_CKPT = "distilbert-base-uncased"
DEFAULT_ENGLISH_CLS = DistilBertForSequenceClassification

register_tokenizer_trans('bert', DEFAULT_ENGLISH_CKPT)


class _NativeEnglishEncoder(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        distilbert_output = self.model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        return hidden_state


class EnglishEncoder(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x: TreeValue):
        assert x.input_ids.shape == x.attention_mask.shape
        pre_shape = x.input_ids.shape[:-1]
        x = x.reshape(-1, x.input_ids.shape[-1])
        output = self.model(x.input_ids, x.attention_mask)
        output = output.reshape(*pre_shape, *output.shape[-2:])
        return output


def create_bert_encoder(ckpt=DEFAULT_ENGLISH_CKPT, cls=DEFAULT_ENGLISH_CLS) -> EnglishEncoder:
    model = cls.from_pretrained(ckpt)
    return EnglishEncoder(_NativeEnglishEncoder(model))


register_encoder('bert', create_bert_encoder)


class BertFineTune(nn.Module):
    def __init__(self, head_n_classes: Mapping[str, int], squeezer: str,
                 mlp_in_featurs: int, mlp_layers: Tuple[int, ...] = (1024,)):
        nn.Module.__init__(self)
        self.encoder = create_encoder('bert')
        self.encoder.requires_grad_(False)
        self.ft = nn.Sequential(
            create_squeezer(squeezer),
            MultiHeadMLP(mlp_in_featurs, head_n_classes, mlp_layers)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.ft(x)
        return x


class BertMeanFineTune(BertFineTune):
    def __init__(self, head_n_classes: Mapping[str, int], align_size: int = 256, emb_size: int = 768):
        BertFineTune.__init__(self, head_n_classes, 'mean', emb_size)


class BertLastFineTune(BertFineTune):
    def __init__(self, head_n_classes: Mapping[str, int], align_size: int = 256, emb_size: int = 768):
        BertFineTune.__init__(self, head_n_classes, 'last', emb_size)


class BertLinearFineTune(BertFineTune):
    def __init__(self, head_n_classes: Mapping[str, int], align_size: int = 256, emb_size: int = 768):
        BertFineTune.__init__(self, head_n_classes, 'linear', emb_size * align_size)


register_model('bert_mean', BertMeanFineTune)
register_model('bert_last', BertLastFineTune)
register_model('bert_linear', BertLinearFineTune)
register_model('bert_mean_512', BertMeanFineTune, align_size=512)
register_model('bert_last_512', BertLastFineTune, align_size=512)
register_model('bert_linear_512', BertLinearFineTune, align_size=512)
