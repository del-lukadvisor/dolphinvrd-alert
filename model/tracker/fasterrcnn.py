from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes

# For all model definition:
# See: https://github.com/pytorch/vision/tree/master/torchvision/models/detection
# and also:  https://pytorch.org/docs/stable/torchvision/models.html

class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes=91):
        backbone = resnet_fpn_backbone(backbone_name='resnet18', pretrained=True) # Another Option could be 'resnet18'/'34'/'50','101'
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, image):
        device = list(self.parameters())[0].device
        image = image.to(device)
        detections = self(image)[0]
        return detections['boxes'].detach(), detections['scores'].detach()

    def get_features(self):
        f = self.features
        pass

    def get_RoI_Pool(self):
        f = self.roi_heads
        print(f)

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores


    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])