from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models._utils import  _ovewrite_value_param
from torchvision.models.detection._utils import  overwrite_eps
from torchvision.ops import misc as misc_nn_ops
from torch import nn
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, _resnet_fpn_extractor
from typing import Any, Optional, TypeVar
import torch

V = TypeVar("V")
# _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
    if actual is not None:
        if actual != expected:
            raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    return expected

def fasterrcnn_resnet50_fpn(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    extend =0,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Faster R-CNN model with a ResNet-50-FPN backbone from the `Faster R-CNN: Towards Real-Time Object
    Detection with Region Proposal Networks <https://arxiv.org/abs/1506.01497>`__
    paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
        :members:
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model

def filter_pred(predicted):
    filtered_predictions = []

    for pred in predicted:
        scores = pred['scores']
        indices = scores > 0.6
        if indices.any():
            filtered_boxes = pred['boxes'][indices]
            filtered_labels = pred['labels'][indices]
            filtered_scores = pred['scores'][indices]
            filtered_pred = {'boxes': filtered_boxes,
                             'labels': filtered_labels,
                             'scores': filtered_scores}
        else:
            filtered_boxes = torch.tensor([[0, 0, 0, 0]])
            filtered_labels = torch.tensor([0])
            filtered_scores =torch.tensor([0])
            filtered_pred = {'boxes': filtered_boxes,
                             'labels': filtered_labels,
                             'scores': filtered_scores}
        filtered_predictions.append(filtered_pred)
    return filtered_predictions

CLASSES = [
           'Akash Bhairav',
            'Bhadrakali Temple',
              'Jalbinayak',
                'Lumadhi Bhadrakali Temple Sankata',
                  'Maitidevi Temple',
                    'Patan Dhoka',
                      'Sano Pashupati',
                        'Swoyambhunath',
                          'Tridevi Temple',
                            'ashok stupa',
                              'birupakshya',
                                'chamunda mai',
                                  'charumati',
                                    'mahadev temple',
                                      'taleju bell_KDS',
                                        'pratappur temple']

classes=['Akash Bhairav','ashok stupa','Badrinath','Bagbairav',
'Balkumari, Bhaktapur',
'BalNilkantha',
'basantapur tower',
'Bhadrakali Temple',
'bhairavnath temple',
'bhaktapur tower','bhimeleshvara',
'Bhimsen Temple','Bhupatindra Malla Column',
'bhuvana lakshmeshvara',
'birupakshya',
'Buddha Statue',
'chakku bakku',
'chamunda mai',
'Chandeshwori Temple',
'Char Narayan Temple',
'charumati',
'chasin dega',
'Chayasilin Mandap',
'Dakshin Barahi',
'degu tale',
'Dharahara',
'Fasidega Temple',
'Garud Statue',
'garud',
'Ghantaghar',
'golden gate',
'golden temple',
'Gopinath krishna Temple',
'guyeshwori',
'hanuman idol',
'Harishankar Temple',
'indrapura',
'Isckon Temple',
'jagannatha temple',
'Jalbinayak',
'Jamachen Monastry',
'jame masjid',
'jaya bageshwori',
'kala-bhairava',
'kasthamandap',
'kavindrapura sattal',
'Kedamatha Tirtha',
'Khumbeshwor mahadev',
'kiranteshwor mahadev',
'kirtipur tower',
'Kotilingeshvara',
'Krishna mandir PDS',
'Krishna_temple _kobahal',
'Kumari Ghar',
'kumaristhan',
'kumbheshwor mahadev',
'lalitpur tower',
'lokeshwor temple bhaktapur',
'Lumadhi Bhadrakali Temple Sankata',
'Mahabauddha Asan',
'mahadev temple',
'Maipi Temple',
'Maitidevi Temple',
'manamaiju temple',
'nagarmandap shree kriti bihar',
'narayan temple',
'National Gallery',
'Naxal Bhagwati',
'Nyatapola temple',
'Palace of 55 Windows',
'Panchamukhi Hanuman',
'Patan Dhoka',
'Pilot Baba',
'PimBahal Gumba',
'pratap malla column',
'pratappur temple',
'Ram Mandir',
'Ranipokhari',
'red gumba',
'sahid gate',
'Sankha Statue',
'Sano Pashupati',
'Santaneshwor Mahadev',
'shantidham',
'Shiva Temple',
'shveta bhairava',
'Siddhi Lakshmi temple',
'simha sattal',
'Swoyambhunath',
'taleju bell pds',
'taleju bell_BDS',
'taleju bell_KDS',
'taleju temple',
'taleju_temple_south',
'trailokya mohan',
'Tridevi Temple',
'uma maheshwor',
'ume_maheshwara',
'Vastala Temple',
'vishnu temple',
'Wakupati Narayan Temple',
'wishing well budhha statue',
'Yetkha Bahal',
'yog_narendra_malla_statue']