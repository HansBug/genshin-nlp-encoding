# genshin-nlp-encoding

NLP Encoder fine-tuning for Genshin Impact

## Install

```shell
git clone https://github.com/HansBug/genshin-nlp-encoding.git
cd genshin-nlp-encoding
pip install -r requirements.txt
```

## Train

```python
from ditk import logging

from ft.train import train

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train(
        'runs/bert_mean_seed_0',
        model_name='bert_mean',
        datasource='annotation.xlsx',
        seed=0,
        max_epochs=50
    )

```

## Infer

```python
import torch
from torch import nn
from treevalue import FastTreeValue

from ft import tokenize
from ft.models import load_model_from_ckpt


class ModelWithEncoding(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        encoded = self.model.encoder(x)
        predicted = self.model.ft(encoded)

        return FastTreeValue({
            'encoded': encoded,
            'prediced': predicted,
        })


_torch_stack = FastTreeValue.func(subside=True)(torch.stack)

if __name__ == '__main__':
    # use your ckpt here!
    m = load_model_from_ckpt('runs/bert_mean_new_seed_0/ckpts/best.ckpt')
    m = ModelWithEncoding(m)
    print(m)

    with torch.no_grad():
        # batch input
        input_ = _torch_stack([
            tokenize(
                """
        "Combat Action: When your active character is Electro Hypostasis, heal that character for 3 HP and attach the Electro Crystal Core to them.
        (You must have Electro Hypostasis in your deck to add this card to your deck.)"
                """, tokenizer_name='bert',
            ),
            tokenize(
                """
        "The character deals +1 DMG.
        When your character triggers an Elemental Reaction: Deal +1 DMG. (Twice per Round)
        (Only Catalyst Characters can equip this. A character can equip a maximum of 1 Weapon)"
                """, tokenizer_name='bert',
            ),
        ])

        print(input_)
        output = m(input_)
        print(output)

```

