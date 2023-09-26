# File  : __init__.py.py
# Author: SunJun
# Date  : 2023/8/30
import json
import os
from collections import OrderedDict

import cn_clip.clip as clip
import torch

from .cn_clip_roberta import CLIPBertWarper


def build_cn_toberta(use_checkpoint=False, pretrain_path=None):
    # here pretrain model is the cn clip model
    tokenizer = clip.tokenize

    bert_config = json.load(
        open(os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                'roberta_base.json'
            )
        )
    )
    bert_config['embed_dim'] = 768
    bert_config['use_checkpoint'] = use_checkpoint
    bert = CLIPBertWarper(**bert_config)
    if pretrain_path is not None:
        assert os.path.exists(pretrain_path), f'{pretrain_path} not exists.'
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        sd = checkpoint["state_dict"]
        # bert_sd = OrderedDict()
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
            # for k, v in sd.items():
            #     if 'bert.pool' not in k:
            #         sd[k[len('module.'):]] = v
        bert.load_state_dict(sd, strict=True)
    return tokenizer, bert
