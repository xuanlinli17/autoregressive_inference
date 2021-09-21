from voi.process.rcnn_features import RCNNFeatures
from torchvision.datasets.folder import default_loader
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.roi_heads import keypointrcnn_inference
from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.ops import boxes as box_ops

from collections import OrderedDict
from multiprocessing import Process
from multiprocessing import set_start_method

import pickle as pkl
import os
import tree

import tensorflow as tf
import torchvision
import torch.nn.functional as F
import torch

import types
import math


def get_faster_rcnn():
    """Creates a modified faster rcnn model for extracting
    intermediate region features

    Returns:

    model: nn.Module
        a torchvision faster rcnn that returns box features
        in addition to box predictions and classes"""

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # standard rcnn does not return box features but ours does
    model.roi_heads.forward = types.MethodType(roi_heads_forward, model.roi_heads)
    model.forward = types.MethodType(rcnn_forward, model)
    return model.cuda()


def get_mask_rcnn():
    """Creates a modified mask rcnn model for extracting
    intermediate region features

    Returns:

    model: nn.Module
        a torchvision faster rcnn that returns box features
        in addition to box predictions and classes"""

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # standard rcnn does not return box features but ours does
    model.roi_heads.forward = types.MethodType(roi_heads_forward, model.roi_heads)
    model.forward = types.MethodType(rcnn_forward, model)
    return model.cuda()


def get_keypoint_rcnn():
    """Creates a modified mask rcnn model for extracting
    intermediate region features

    Returns:

    model: nn.Module
        a torchvision faster rcnn that returns box features
        in addition to box predictions and classes"""

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # standard rcnn does not return box features but ours does
    model.roi_heads.forward = types.MethodType(roi_heads_forward, model.roi_heads)
    model.forward = types.MethodType(rcnn_forward, model)
    return model.cuda()


def roi_heads_forward(self,
                      features,
                      proposals,
                      image_shapes,
                      targets=None):
    """Hack into the torchvision model to obtain features for
    training caption model; training is assumed to be false

    https://github.com/pytorch/vision/blob/master/
        torchvision/models/detection/roi_heads.py"""

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])

    boxes, scores, labels, box_features = roi_postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes,
        box_features)
    num_images = len(boxes)
    for i in range(num_images):
        result.append({
            "boxes_features": box_features[i],
            "boxes": boxes[i],
            "labels": labels[i],
            "scores": scores[i]})

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features,
                                               mask_proposals,
                                               image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

        else:
            mask_logits = torch.tensor(0)
            raise Exception("Expected mask_roi_pool to be not None")

        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            r["masks_features"] = mask_features
            r["masks"] = mask_prob

    # keep none checks in if conditional so torchscript will conditionally
    # compile each branch
    if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
            and self.keypoint_predictor is not None:
        keypoint_proposals = [p["boxes"] for p in result]

        keypoint_features = self.keypoint_roi_pool(features,
                                                   keypoint_proposals,
                                                   image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        assert keypoint_logits is not None
        assert keypoint_proposals is not None

        keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits,
                                                            keypoint_proposals)
        for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
            r["keypoints_features"] = keypoint_features
            r["keypoints"] = keypoint_prob
            r["keypoints_scores"] = kps

    return result, dict()


def roi_postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes,
                               *extra_tensors):
    """Hack into the torchvision model to obtain features for
    training caption model; training is assumed to be false

    https://github.com/pytorch/vision/blob/master/
        torchvision/models/detection/roi_heads.py"""

    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    # split boxes and scores per image
    if len(boxes_per_image) == 1:
        # TODO : remove this when ONNX support dynamic split sizes
        # and just assign to pred_boxes instead of pred_boxes_list
        pred_boxes_list = [pred_boxes]
        pred_scores_list = [pred_scores]
        extra_tensors_list = [[x] for x in extra_tensors]
    else:
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        extra_tensors_list = [x.split(boxes_per_image, 0) for x in extra_tensors]

    all_boxes = []
    all_scores = []
    all_labels = []
    all_extras = [[] for _ in extra_tensors]
    for boxes, scores, image_shape, *extras in zip(
            pred_boxes_list, pred_scores_list, image_shapes, *extra_tensors_list):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        # each feature vector is used for all 91 class predictions
        # there are 90 classes (minus the background)
        # take the feature vector corresponding to each class
        extras = [x[inds // (num_classes - 1)] for x in extras]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        # each feature vector is used for all 91 class predictions
        # there are 90 classes (minus the background)
        # take the feature vector corresponding to each class
        extras = [x[keep] for x in extras]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        # each feature vector is used for all 91 class predictions
        # there are 90 classes (minus the background)
        # take the feature vector corresponding to each class
        extras = [x[keep] for x in extras]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        for x, y in zip(all_extras, extras):
            x.append(y)

    return [all_boxes, all_scores, all_labels, *all_extras]


def rcnn_forward(self,
                 images,
                 targets=None):
    """Hack into the torchvision model to obtain features for
    training caption model; training is assumed to be false

    https://github.com/pytorch/vision/blob/master/
        torchvision/models/detection/generalized_rcnn.py"""

    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)
    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])

    proposals, proposal_losses = self.rpn(images, features, targets)
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    return [dict(features=tree.map_structure(
        lambda x: x[i], features), **d) for i, d in enumerate(detections)]


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_paths):
        """Creates a torch data set for loading images from the disk
        and extracting features

        Arguments:

        root_dir: string:
            Directory with all the images.
        transform: callable, optional:
            Optional transform to be applied on a sample"""

        self.paths = image_paths
        self.t = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        return self.t(default_loader(image_path)), image_path


def to_list(batch):
    return zip(*batch)


def process_images_backend(gpu_id,
                           out_feature_folder,
                           paths,
                           batch_size):
    """Process images in a specified folder from a standard format
    into numpy features using torchvision

    Arguments:

    gpu_id: int
        the index of the gpu to use in this process
    out_feature_folder: str
        the path to a new folder where image features
        will be placed on the disk
    paths: list
        a list of file names to be processed
    batch_size: int
        the number of images to process in parallel"""

    with torch.cuda.device(gpu_id):

        # create a faster rcnn instance to extract features
        model = get_faster_rcnn()
        dataloader = torch.utils.data.DataLoader(Dataset(paths),
                                                 batch_size=batch_size,
                                                 num_workers=8,
                                                 collate_fn=to_list)

        # extract images in a batch of images
        for image, p in dataloader:
            with torch.no_grad():
                batch_dict = model(
                    tree.map_structure(lambda x: x.cuda(), image))

            # save region features to disk
            for i, (data, path) in enumerate(zip(batch_dict, p)):

                # calculate a global feature for each image
                pooled_features = tree.map_structure(
                    lambda x: torch.mean(x, dim=[1, 2]),
                    data['features'])
                global_features = torch.cat(
                    tree.flatten(pooled_features), dim=0)

                # normalize box coordinates to be in [0, 1]
                boxes = data['boxes'] / torch.tensor([[
                    image[i].shape[2],
                    image[i].shape[1],
                    image[i].shape[2],
                    image[i].shape[1]]]).cuda()

                # add all features to storage
                samples = RCNNFeatures(
                    global_features=global_features.cpu().detach().numpy(),
                    boxes_features=data['boxes_features'].cpu().detach().numpy(),
                    boxes=boxes.cpu().detach().numpy(),
                    labels=data['labels'].cpu().detach().numpy(),
                    scores=data['scores'].cpu().detach().numpy())

                # write a named tuple to the disk containing features
                sample_path = os.path.join(
                    out_feature_folder, os.path.basename(path) + '.pkl')
                with tf.io.gfile.GFile(sample_path, "wb") as f:
                    f.write(pkl.dumps(samples))


def process_images(out_feature_folder: str,
                   in_folder: str,
                   batch_size: int):
    """Process images in a specified folder from a standard format
    into numpy features using torchvision

    Arguments:

    out_feature_folder: str
        the path to a new folder where image features
        will be placed on the disk
    in_folder: str
        a folder that contains image files
    batch_size: int
        the number of images to process in parallel"""

    set_start_method('spawn')

    # create the output folder if it does not exist
    tf.io.gfile.makedirs(out_feature_folder)

    # make a list all images in the in directory
    paths = tf.io.gfile.glob(os.path.join(in_folder, "*.jpg"))
    paths = [p for p in paths if not tf.io.gfile.exists(
        os.path.join(out_feature_folder,
                     os.path.basename(p) + '.pkl'))]

    num_gpu = torch.cuda.device_count()
    split = math.ceil(len(paths) / num_gpu)
    processes = [
        Process(target=process_images_backend,
                args=(i,
                      out_feature_folder,
                      paths[i*split:(i + 1)*split],
                      batch_size)) for i in range(num_gpu)]

    # distribute inference to all available gpus
    for p in processes:
        p.start()
    for p in processes:
        p.join()

