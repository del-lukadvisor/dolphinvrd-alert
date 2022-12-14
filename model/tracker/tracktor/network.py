from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes


def resize_boxes_v2(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)
        detections = self(img)[0]
        return detections['boxes'].detach(), detections['scores'].detach()

    # need edit
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

        pred_class = torch.argmax(pred_scores, dim=1)

        # This is the part that need modification

        ''' Original Code
        pred_boxes = pred_boxes[:, -1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, -1:].squeeze(dim=1).detach()
        '''        
        '''
        print(pred_boxes[:, -1:].shape)
        print(pred_boxes[:, 1:1 + 1].shape)
        '''

        ''' #Running Code for Dolphin
        print(pred_boxes.shape);
        print(pred_boxes[:, 1:2, :].shape);
        print(pred_boxes[:, 1:2].squeeze(dim=1).shape)        
        print(torch.equal(pred_boxes[:, 1:2, :].squeeze(dim=1), pred_boxes[:, 1, :]))
        print(pred_class)
        '''

        ''' Should be working
        final_pred_box = torch.zeros(len(pred_boxes), 4)
        for i,(pb, cls) in enumerate(zip(pred_boxes, pred_class)):    
            final_pred_box[i, :] = pb[cls]
        final_pred_box = final_pred_box.to(pred_boxes.device)
        pred_boxes = final_pred_box
        '''

        pred_boxes = pred_boxes[:, 1:2].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:2].squeeze(dim=1).detach()

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




if __name__ == '__main__':
    test_Tracktor = FRCNN_FPN(num_classes=3)
    test_Tracktor.load()


