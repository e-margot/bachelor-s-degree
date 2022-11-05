import os

from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple, List
import torchvision
from PIL import Image
import torch
from my_coco import MyCOCO


class AugMyCocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.coco = MyCOCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, os.path.basename(path))).convert('L').resize((256, 256))
        im_aug = torchvision.transforms.RandomInvert(p=0.67)(image)
        im_aug = torchvision.transforms.functional.adjust_sharpness(im_aug, 10)
        return im_aug

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = torch.as_tensor(self._load_target(id), dtype=torch.float32)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    def printf(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        im = Image.open(os.path.join(self.root, os.path.basename(path))).convert('L').resize((256, 256))
        im_aug = torchvision.transforms.functional.adjust_sharpness(im, 15)
