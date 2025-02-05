# Focoos Foundational Models

With the Focoos SDK, you can take advantage of a collection of foundational models that are optimized for a range of computer vision tasks. These pre-trained models, covering detection and semantic segmentation across various domains, provide an excellent starting point for your specific use case. Whether you need to fine-tune for custom requirements or adapt them to your application, these models offer a solid foundation to accelerate your development process.

---

## Semantic Segmentation Models

| Model Name | Architecture | Domain (Classes) | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|------------------|----------|---------|--------------|
| [fai-m2f-l-ade](models/fai-m2f-l-ade.md) | [Mask2Former](https://github.com/facebookresearch/Mask2Former) ([Resnet-101](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)) | Common Scene (150) | [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | mIoU: 48.27<br>mAcc: 62.15 | 73 |
| [fai-m2f-m-ade](models/fai-m2f-m-ade.md) | [Mask2Former](https://github.com/facebookresearch/Mask2Former) ([STDC-2](https://github.com/MichaelFan01/STDC-Seg)) | Common Scene (150) | [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | mIoU: 45.32<br>mACC: 57.75 | 127 |
| [fai-m2f-s-ade](models/fai-m2f-s-ade.md) | [Mask2Former](https://github.com/facebookresearch/Mask2Former) ([STDC-1](https://github.com/MichaelFan01/STDC-Seg)) | Common Scene (150) | [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | mIoU: 41.23<br>mAcc: 52.21 | 189 |

## Object Detection Models

| Model Name | Architecture | Domain (Classes) | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|------------------|----------|---------|--------------|
| [fai-rtdetr-l-coco](models/fai-rtdetr-l-coco.md) | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) ([Resnet-50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | bbox/AP: 53.06<br>bbox/AP50: 70.91 | 87 |
| [fai-rtdetr-m-coco](models/fai-rtdetr-m-coco.md) | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) ([STDC-2](https://github.com/MichaelFan01/STDC-Seg)) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | bbox/AP: 44.69<br>bbox/AP50: 61.63 | 181 |
| [fai-rtdetr-s-coco](models/fai-rtdetr-s-coco.md) | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) ([STDC-1](https://github.com/MichaelFan01/STDC-Seg)) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | bbox/AP: 42.58<br>bbox/AP50: 59.22 | 220 |
| [fai-rtdetr-n-coco](models/fai-rtdetr-n-coco.md) | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) ([STDC-1](https://github.com/MichaelFan01/STDC-Seg)) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | bbox/AP: 40.59<br>bbox/AP50: 56.69 | 269 |
| [fai-rtdetr-m-obj365](models/fai-rtdetr-m-obj365.md) | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) ([Resnet50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)) | Common Objects (365) | [Objects365](https://www.objects365.org/overview.html) | bbox/AP: 34.60<br>bbox/AP50: 45.81 | 87 |

## Instance Segmentation Models

| Model Name | Architecture | Domain (Classes) | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|------------------|----------|---------|--------------|
| [fai-m2f-l-coco-ins](models/fai-m2f-l-coco-ins.md) | [Mask2Former](https://github.com/facebookresearch/Mask2Former) ([Resnet-50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | segm/AP: 42.39<br>segm/AP50: 66.12 | 54 |

Note:
<br> mIoU = Intersection over Union averaged by class </br>
<br> mAcc = Pixel Accuracy averaged by class </br>
<br> AP = Average Precision averaged by class </br>
<br> AP50 = Average Precision at IoU threshold 0.50 averaged by class </br>
<br> FPS = Frames per second computed using TensorRT with resolution 640x640 </br>
