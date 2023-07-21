import torch.nn.functional as F
from transformers import AutoTokenizer
from treevalue import FastTreeValue

_TOKENIZERS = {}


def register_tokenizer(name, tokenize_func):
    _TOKENIZERS[name] = tokenize_func


def register_tokenizer_trans(name, pretrained_model_name_or_path, *inputs, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    def _tokenize_func(sentence):
        return tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    register_tokenizer(name, _tokenize_func)


def tokenize(sentence, tokenizer_name: str, align_size: int = 64):
    tokens = _TOKENIZERS[tokenizer_name](sentence)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    token_type_ids = tokens.get('token_type_ids', None)

    _r_b, _r_size = input_ids.shape
    assert _r_b == 1, f'Batch should be 1, but {_r_b!r} found.'

    padded_tokens = input_ids
    if _r_size < align_size:
        padded_tokens = F.pad(padded_tokens, (0, align_size - _r_size), mode='constant', value=0)
    padded_tokens = padded_tokens[:, :align_size].squeeze(0)

    padded_mask = attention_mask
    if _r_size < align_size:
        padded_mask = F.pad(padded_mask, (0, align_size - _r_size), mode='constant', value=0)
    padded_mask = padded_mask[:, :align_size].squeeze(0)

    if token_type_ids is not None:
        padded_type_ids = token_type_ids
        if _r_size < align_size:
            padded_type_ids = F.pad(padded_type_ids, (0, align_size - _r_size), mode='constant', value=0)
        padded_type_ids = padded_type_ids[:, :align_size].squeeze(0)
    else:
        padded_type_ids = None

    if padded_type_ids is not None:
        return FastTreeValue({
            'input_ids': padded_tokens,
            'attention_mask': padded_mask,
            'token_type_ids': padded_type_ids,
        })
    else:
        return FastTreeValue({
            'input_ids': padded_tokens,
            'attention_mask': padded_mask,
        })
