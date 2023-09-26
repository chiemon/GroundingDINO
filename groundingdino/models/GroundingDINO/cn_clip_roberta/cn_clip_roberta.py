# File  : cn_clip_roberta.py
# Author: SunJun
# Date  : 2023/8/30
import numpy as np
import torch
import torch.nn as nn
from cn_clip.clip import _tokenizer
from cn_clip.clip.configuration_bert import BertConfig
from cn_clip.clip.modeling_bert import BertModel


class CLIPBertWarper(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float,
                 text_hidden_act: str,
                 text_hidden_dropout_prob: float,
                 text_hidden_size: int,
                 text_initializer_range: float,
                 text_intermediate_size: int,
                 text_max_position_embeddings: int,
                 text_num_attention_heads: int,
                 text_num_hidden_layers: int,
                 text_type_vocab_size: int,
                 tokenizer=_tokenizer,
                 use_checkpoint=False,
                 # vision head width, added this param for ViT-H
                 ):
        super().__init__()
        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.embed_dim = embed_dim
        self.bert = BertModel(self.bert_config)
        self.bert.encoder.grad_checkpointing = use_checkpoint
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        self.initialize_parameters()

    def initialize_parameters(self):
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.bert.set_grad_checkpointing(enable)

    def forward(self, text, dtype):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(dtype)  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection
