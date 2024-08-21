import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from functools import singledispatchmethod
from typing import Any, Dict, Optional, List, Tuple, Union
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode

class InnerRandomCrop(v2.Transform):

    def __init__(self, size_w, size_h):
        super().__init__()

        self.crop_size_w = size_w
        self.crop_size_h = size_h


    @singledispatchmethod
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Default Behavior: Don't modify the input"""
        return inpt

    @_transform.register(torch.Tensor)
    @_transform.register(tv_tensors.Image)
    def _(self, inpt: Union[torch.Tensor, tv_tensors.Image], params: Dict[str, Any]) -> Any:
        """Apply the inner crop to the image"""
        # Randomly crop inside the image
        self.img_h, self.img_w = inpt.shape[-2:]

        if self.img_h - self.crop_size_h > 0:
            self.crop_center_h = torch.randint(0, self.img_h - self.crop_size_h, (1,))
        else:
            self.crop_center_h = 0
        if self.img_w - self.crop_size_w > 0:
            self.crop_center_w = torch.randint(0, self.img_w - self.crop_size_w, (1,))
        else:
            self.crop_center_w = 0

        crop_img = inpt[..., self.crop_center_h :self.crop_center_h + self.crop_size_h,
                        self.crop_center_w:self.crop_center_w + self.crop_size_w ]

        return crop_img

    @_transform.register(Mask)
    def _(self, inpt: Mask, params: Dict[str, Any]) -> Any:

        assert self.img_h == inpt.shape[0] and self.img_w == inpt.shape[1], "Image and mask should have the same shape"

        crop_mask = inpt[..., self.crop_center_h :self.crop_center_h + self.crop_size_h,
                        self.crop_center_w:self.crop_center_w + self.crop_size_w ]

        return Mask(crop_mask)


class Resize(v2.Transform):

    def __init__(self, size_w, size_h):
        super().__init__()

        self.size_w = size_w
        self.size_h = size_h


    @singledispatchmethod
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Default Behavior: Don't modify the input"""
        return inpt

    @_transform.register(torch.Tensor)
    @_transform.register(tv_tensors.Image)
    def _(self, inpt: Union[torch.Tensor, tv_tensors.Image], params: Dict[str, Any]) -> Any:
        """Apply the inner crop to the image"""
        # resize the image
        resized_img = v2.functional.resize(inpt, (self.size_h, self.size_w), interpolation=InterpolationMode.BILINEAR)
        return resized_img

    #@_transform.register(BoundingBoxes)
    @_transform.register(Mask)
    def _(self, inpt: Mask, params: Dict[str, Any]) -> Any:
        #resize mask
        resized_mask = v2.functional.resize(inpt, (self.size_h, self.size_w), interpolation=InterpolationMode.NEAREST)
        return Mask(resized_mask)



