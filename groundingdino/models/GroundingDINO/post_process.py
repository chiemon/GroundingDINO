import torch
import torch.nn as nn
from torchvision.ops.boxes import nms
from groundingdino.util import box_ops

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    def reformat_pred_class_logit(self, pred_logits, cate_to_token_mask_list):
        """
        Args:
            pred_logits: B, N, T
            text_self_attention: len([ torch.Tensor(c, T)]) = B

        Returns:
            len([torch.Tensor(N, c)]) = B
        """
        reformat_pred_logits = []
        for bidx, cate_to_token_mask in enumerate(cate_to_token_mask_list):
            # remove all False mask
            if cate_to_token_mask[-1].float().sum() == 0.:
                clean_cate_to_token_mask = cate_to_token_mask[:-1]
            else:
                clean_cate_to_token_mask = cate_to_token_mask
            assert clean_cate_to_token_mask.ndim == 2, \
                f"expected ndim==2, but got {clean_cate_to_token_mask.ndim}"
            pred_logit = pred_logits[bidx][:, :clean_cate_to_token_mask.shape[1]]  # N, T
            refotmat_pred_logit = (pred_logit.unsqueeze(0) * \
                                   clean_cate_to_token_mask.unsqueeze(1)).sum(dim=-1).T  # c, N
            refotmat_pred_logit /= cate_to_token_mask.float().sum(dim=-1)  # mean logits to compute cost
            reformat_pred_logits.append(refotmat_pred_logit)  # different batch have different gts

        return reformat_pred_logits

    @torch.no_grad()
    def forward(self, outputs, ori_captions, image_ids, not_to_xyxy=False, test=False,
                use_absolute_coordinates=False, target_sizes=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            ori_captions: List[List[str]]
            image_ids: List[int]
            use_absolute_coordinates: if set True boxes from relative [0, 1] to absolute [0, height] coordinates.
                        if use_absolute_coordinates=True, target_sizes must be not None
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        cate_to_token_mask_list = outputs['text_dict']['cate_to_token_mask_list']
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_iou = outputs.get('pred_ious', None) # B, N, 1
        out_logits = self.reformat_pred_class_logit(out_logits, cate_to_token_mask_list)  # B, N, C
        assert len(out_logits) == 1
        out_logits = torch.stack(out_logits)  # 1, N, C
        if use_absolute_coordinates:
            assert target_sizes != None
            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2
            assert target_sizes.shape[0] == 1, 'set batchSize to 1.'

        prob = out_logits.sigmoid()  # B, N, 4
        if out_iou is None:
            out_iou = torch.ones_like(prob)
        prob = prob * out_iou # rescore by iou
        image_ids = torch.as_tensor(image_ids).reshape(-1, 1, 1).repeat(1, prob.shape[1], prob.shape[2])
        image_ids = image_ids.view(out_logits.shape[0], -1).to(prob.device)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        image_ids = image_ids[torch.arange(out_logits.shape[0]), topk_indexes]
        assert image_ids.size() == labels.size()
        # transfer labels to texts
        texts = []
        for batch_idx, label in enumerate(labels):
            text = [ori_captions[batch_idx][lidx] for lidx in label]
            texts.append(text)

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        if use_absolute_coordinates:
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b, s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i], 'texts': t[i], 'image_ids': image_ids[i]} for
                       s, l, b, t, i in
                       zip(scores, labels, boxes, texts, image_ids, item_indices)]
        else:
            results = [
                {'scores': s, 'labels': l, 'boxes': b, 'texts': t, 'image_ids': d}
                for s, l, b, t, d in
                zip(scores, labels, boxes, texts, image_ids)
            ]

        return results

    def format_results(self, results):
        # this format used to evaluate
        format_boxes, format_scores, format_texts, format_image_id = [], [], [], []
        for result in results:
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']  # x, y, x, y
            # boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2] # x, y, x, y
            texts = result['texts']
            image_ids = result['image_ids']
            format_boxes.append(boxes)
            format_scores.append(scores)
            format_texts.append(texts)
            format_image_id.append(image_ids[0])
        return dict(
            boxes=format_boxes,
            scores=format_scores,
            texts=format_texts,
            image_id=format_image_id,
        )