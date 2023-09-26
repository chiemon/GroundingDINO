# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import os
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        use_pre_text_embeddings=False,
        pre_text_embeddings_path=None,
        dec_pred_iou_embed_share=True,
        use_iou_aware=False,
        use_cn_clip_bert=False,
        cn_clip_pretrain_path=None,
        cn_clip_use_checkpoint=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        global feat_map_in_channel
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present
        self.use_iou_aware = use_iou_aware
        self.use_pre_text_embeddings = use_pre_text_embeddings

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.use_cn_clip_bert = use_cn_clip_bert
        if not use_pre_text_embeddings:
            if not use_cn_clip_bert:
                self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
                self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
                self.bert.pooler.dense.weight.requires_grad_(False)
                self.bert.pooler.dense.bias.requires_grad_(False)
                self.bert = BertModelWarper(bert_model=self.bert)
                feat_map_in_channel = self.bert.config.hidden_size
            else:
                from .cn_clip_roberta import build_cn_toberta
                self.tokenizer, self.bert = build_cn_toberta(
                    use_checkpoint=cn_clip_use_checkpoint,
                    pretrain_path=cn_clip_pretrain_path
                )
                feat_map_in_channel = self.bert.embed_dim
                vectors = torch.empty(feat_map_in_channel, dtype=torch.float32)
                nn.init.normal_(vectors, std=0.02)
                self.padding = nn.Parameter(vectors, )
        else:
            assert not self.use_cn_clip_bert, f'Please verify whether use cn_clip_bert.'
            self.tokenizer, self.bert = None, None

            assert os.path.exists(pre_text_embeddings_path)
            text2feat = torch.load(pre_text_embeddings_path, map_location='cpu')
            for k, v in text2feat.items():
                self.register_buffer(k, v.type(dtype=torch.float32))  # register embeddings
                feat_map_in_channel = v.shape[0]  # 768
            vectors = torch.empty(feat_map_in_channel, dtype=torch.float32)
            nn.init.normal_(vectors, std=0.02)
            self.padding = nn.Parameter(vectors, )

        self.feat_map = nn.Linear(feat_map_in_channel, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]) if (not use_pre_text_embeddings and not use_cn_clip_bert) else None

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.dec_pred_iou_embed_share = dec_pred_iou_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if self.use_iou_aware:
            _iou_embed = MLP(hidden_dim, hidden_dim, 1, 2)
            nn.init.constant_(_iou_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_iou_embed.layers[-1].bias.data, 0)
            if dec_pred_iou_embed_share:
                iou_embed_layerlist = [_iou_embed for i in range(transformer.num_decoder_layers)]
            else:
                iou_embed_layerlist = [
                    copy.deepcopy(_iou_embed) for i in range(transformer.num_decoder_layers)
                ]

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.iou_embed = nn.ModuleList(iou_embed_layerlist) if self.use_iou_aware else None
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def extract_text_feat_by_cn_bert(self, samples, targets: List, **kw):
        embeddings = []
        if targets is None:
            captions = kw.get('captions')
        else:
            captions = [target['ori_caption'] for target in targets]

        max_len = max([len(cap) for cap in captions])
        batch_size = len(captions)
        if self.training:
            max_len += 1
        text_self_attention_masks = (
            torch.eye(max_len, device=samples.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
        )
        cate_to_token_mask_list = []
        text_token_mask = torch.ones((batch_size, max_len), dtype=text_self_attention_masks.dtype,
                                     device=samples.device)
        for bidx, caption in enumerate(captions):
            text_tokens = self.tokenizer(caption).to(samples.device)
            embedding = self.bert(text_tokens, dtype=self.feat_map.weight.dtype) # N * C
            text_self_attention_masks[bidx, len(embedding):, len(embedding):] = True
            c2t_maski = torch.eye(max_len, device=samples.device).bool()
            c2t_maski[len(embedding):, len(embedding):] = False
            text_token_mask[bidx, len(embedding):] = 0.  # padding mask to 0
            cate_to_token_mask_list.append(c2t_maski)
            padding_embedding = []
            for _ in range(max_len - embedding.shape[0]):
                padding_embedding.append(self.padding)  # pad to same length
            if padding_embedding:
                padding_embedding = torch.stack(padding_embedding)
                embedding = torch.cat([embedding, padding_embedding], dim=0) # N, C
            assert embedding.shape[0] == max_len
            embeddings.append(embedding)
        encoded_text = torch.stack(embeddings)  # B, N, C
        encoded_text = self.feat_map(encoded_text)  # bs, 195, d_model
        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "position_ids": torch.zeros((len(captions), max_len), device=samples.device),
            "text_token_mask": text_token_mask.bool(),
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            "cate_to_token_mask_list": cate_to_token_mask_list,
        }
        return encoded_text, text_dict

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if len(self.backbone.num_channels) != 4:
            if targets is None:
                captions = kw["captions"]
            else:
                captions = [t["caption"] for t in targets]

            # encoder texts
            tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
                samples.device
            )
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

            if text_self_attention_masks.shape[1] > self.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            # extract text embeddings
            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                # import ipdb; ipdb.set_trace()
                tokenized_for_encoder = tokenized

            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

            encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]

            text_dict = {
                "encoded_text": encoded_text,  # bs, 195, d_model
                "text_token_mask": text_token_mask,  # bs, 195
                "position_ids": position_ids,  # bs, 195
                "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
            }

            # import ipdb; ipdb.set_trace()
        else:
            assert targets is None
            if self.bert is not None:
                if not self.use_cn_clip_bert:
                    (bert_output, tokenized, position_ids, text_self_attention_masks,
                            cate_to_token_mask_list) = self.extract_text_feat_by_bert(samples, targets, **kw)
                    encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
                    text_token_mask = tokenized.attention_mask.bool()  # bs, 195
                    # text_token_mask: True for nomask, False for mask
                    # text_self_attention_masks: True for nomask, False for mask
                    if encoded_text.shape[1] > self.max_text_len:
                        encoded_text = encoded_text[:, : self.max_text_len, :]
                        text_token_mask = text_token_mask[:, : self.max_text_len]
                        position_ids = position_ids[:, : self.max_text_len]
                        text_self_attention_masks = text_self_attention_masks[
                                                    :, : self.max_text_len, : self.max_text_len
                                                    ]
                    text_dict = {
                        "encoded_text": encoded_text,  # bs, 195, d_model
                        "text_token_mask": text_token_mask,  # bs, 195
                        "position_ids": position_ids,  # bs, 195
                        "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
                        "cate_to_token_mask_list": cate_to_token_mask_list,
                        # bs = len([torch.Tensor(c1, 195), torch.Tensor(c2, 195) ...])
                    }
                else:
                    encoded_text, text_dict = self.extract_text_feat_by_cn_bert(samples=samples, targets=targets, **kw)
            else:
                captions = kw.get('captions')
                assert len(captions) == 1
                embeddings = []
                max_len = len(captions[0])
                text_self_attention_masks = (
                    torch.eye(max_len, device=samples.device).bool().unsqueeze(0).repeat(len(captions), 1, 1)
                )
                cate_to_token_mask_list = []
                text_token_mask = torch.ones((len(captions), max_len), dtype=text_self_attention_masks.dtype,
                                            device=samples.device)

                embedding = [getattr(self, ori_cap) for ori_cap in captions[0]]
                text_self_attention_masks[0, len(embedding):, len(embedding):] = True
                c2t_maski = torch.eye(max_len, device=samples.device).bool()
                c2t_maski[len(embedding):, len(embedding):] = False
                text_token_mask[0, len(embedding):] = 0.  # padding mask to 0
                cate_to_token_mask_list.append(c2t_maski)

                embedding = torch.stack(embedding, dim=0)
                padding_embedding = []
                for _ in range(max_len - embedding.shape[0]):
                    padding_embedding.append(self.padding)  # pad to same length

                if padding_embedding:
                    padding_embedding = torch.stack(padding_embedding)
                    embedding = torch.concat([embedding, padding_embedding], dim=0)  # N, C

                assert embedding.shape[0] == max_len
                embeddings.append(embedding)

                encoded_text = torch.stack(embeddings)  # B, N, C
                encoded_text = self.feat_map(encoded_text)  # bs, 195, d_model

                text_dict = {
                    "encoded_text": encoded_text,  # bs, 195, d_model
                    "position_ids": torch.zeros((len(captions), max_len), device=samples.device),
                    "text_token_mask": text_token_mask.bool(),
                    "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
                    "cate_to_token_mask_list": cate_to_token_mask_list,
                }
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, self.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        if self.use_iou_aware:
            outputs_iou_list = [
                layer_iou_embd(layer_hs) for layer_iou_embd, layer_hs in zip(self.iou_embed, hs)
            ]
            outputs_iou_list = torch.stack(outputs_iou_list)
        else:
            outputs_iou_list = None

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
            "pred_ious": outputs_iou_list[-1] if outputs_iou_list is not None else None,
            "text_dict": text_dict
        }

        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() ## If necessary
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.use_iou_aware:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_ious": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_iou[:-1])
            ]
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    use_pre_text_embeddings = getattr(args, 'use_pre_text_embeddings', False)
    pre_text_embeddings_path = getattr(args, 'pre_text_embeddings_path', None)
    dec_pred_iou_embed_share = getattr(args, 'dec_pred_iou_embed_share', True)
    use_iou_aware = getattr(args, 'use_iou_aware', False)
    use_cn_clip_bert = getattr(args, 'use_cn_clip_bert', False)
    cn_clip_pretrain_path = getattr(args, 'cn_clip_pretrain_path', None)
    cn_clip_use_checkpoint = getattr(args, 'cn_clip_use_checkpoint', False)

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        use_pre_text_embeddings=use_pre_text_embeddings,
        pre_text_embeddings_path=pre_text_embeddings_path,
        dec_pred_iou_embed_share=dec_pred_iou_embed_share,
        use_iou_aware=use_iou_aware,
        use_cn_clip_bert=use_cn_clip_bert,
        cn_clip_pretrain_path=cn_clip_pretrain_path,
        cn_clip_use_checkpoint=cn_clip_use_checkpoint,
    )

    return model
