import os
from typing import Tuple, List, Optional
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import natsort


class FloodDataset(Dataset):
    """
    Dataset for flood segmentation.
    Just a normal dataset class with some additional methods for L-CRP:
        - class_names;
        - reverse_augmentation and reverse_normalization.
    """
    class_names = ["background", "flood"]

    # ImageNet stats
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        image_size: Tuple[int, int] = (1280, 720),  # (H, W)
        strict_pairing: bool = False,
        mask_suffix_patterns: Optional[List[str]] = None,
    ):
        # directories
        self.image_dir = os.path.join(root_dir, "RGB", split, "JPEG")
        self.mask_dir  = os.path.join(root_dir, "annotations", split, "JPEG")

        # external hook (kept)
        self.transform = transform

        # size (kept)
        self.image_size = image_size  # (H, W)

        # pairing config
        self.strict_pairing = strict_pairing
        self.mask_suffix_patterns = mask_suffix_patterns or [
            r"_mask$", r"-mask$", r"_label$", r"-label$", r"_gt$", r"-gt$", r"_ann$", r"-ann$", r"Ids$"
        ]
        self._mask_suffix_re = re.compile("|".join(self.mask_suffix_patterns), flags=re.IGNORECASE)

        # gather files (non-recursive)
        img_exts  = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
        mask_exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")

        image_files_all = [Path(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(img_exts)]
        mask_files_all  = [Path(self.mask_dir,  f) for f in os.listdir(self.mask_dir)  if f.endswith(mask_exts)]
        image_files_all = natsort.natsorted(image_files_all)
        mask_files_all  = natsort.natsorted(mask_files_all)

        def stem_no_ext(p: Path) -> str: return p.stem
        def norm_mask_stem(s: str) -> str: return self._mask_suffix_re.sub("", s)

        img_by_stem  = {stem_no_ext(p): p for p in image_files_all}
        if self.strict_pairing:
            mask_by_stem = {stem_no_ext(p): p for p in mask_files_all}
        else:
            mask_by_stem = {norm_mask_stem(stem_no_ext(p)): p for p in mask_files_all}

        common = [s for s in natsort.natsorted(img_by_stem.keys()) if s in mask_by_stem]
        self.image_files: List[Path] = [img_by_stem[s]  for s in common]
        self.mask_files:  List[Path] = [mask_by_stem[s] for s in common]

        if (len(self.image_files) == 0 or len(self.mask_files) == 0) and \
           (len(image_files_all) == len(mask_files_all) and len(image_files_all) > 0):
            self.image_files = image_files_all
            self.mask_files  = mask_files_all

        assert len(self.image_files) == len(self.mask_files), "Mismatch between number of images and masks after alignment!"

        # core transforms
        self.resize_img  = transforms.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR)
        self.to_tensor   = transforms.ToTensor()
        self.normalize   = transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        self.resize_mask = transforms.Resize(self.image_size, interpolation=InterpolationMode.NEAREST)

        # reverse_normalization (now returns float32 in [0,1] — SAFE for requires_grad)
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

        def _rev_norm(x: torch.Tensor) -> torch.Tensor:
            """
            Undo ImageNet normalization and return float32 in [0,1].
            This avoids uint8 outputs that would break `.requires_grad_()`.
            """
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if not x.dtype.is_floating_point:
                # Handle uint8 or longs coming from external callers
                x = x.float()
                if x.max().item() > 1.5:  # assume [0..255]
                    x = x / 255.0
            x = (x * std) + mean
            return x.clamp(0.0, 1.0).to(torch.float32, copy=False)

        self.reverse_normalization = _rev_norm

        # detect if user transform is PIL-space (contains ToTensor)
        self._transform_is_pil = False
        if self.transform is not None and hasattr(self.transform, "transforms"):
            self._transform_is_pil = any(isinstance(t, transforms.ToTensor) for t in self.transform.transforms)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        n = len(self.image_files)
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} is out of range for dataset of length {n}.")

        img_path  = self.image_files[idx]
        mask_path = self.mask_files[idx]

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # mask: resize(nearest) + binarize -> LongTensor
        mask_resized = self.resize_mask(mask)
        mask_tensor  = self._mask_to_long01(mask_resized)

        # image: robust handling of user transform
        img_resized = self.resize_img(img)

        if self.transform is None:
            image_tensor = self.normalize(self.to_tensor(img_resized))
        elif self._transform_is_pil:
            out = self.transform(img_resized)
            if isinstance(out, torch.Tensor):
                mx = float(out.max().item()) if out.numel() else 1.0
                if mx <= 1.5:
                    image_tensor = self.normalize(out.float())
                else:
                    image_tensor = self.normalize(out.float().div_(255.0).clamp_(0, 1))
            else:
                image_tensor = self.normalize(self.to_tensor(out))
        else:
            image_tensor = self.normalize(self.to_tensor(img_resized))
            image_tensor = self.transform(image_tensor)

        # ensure float32 for autograd safety
        image_tensor = image_tensor.to(torch.float32, copy=False)

        return image_tensor, mask_tensor

    # --- unprocessed float32 (no normalization), for CRP/PCX if needed ---
    def get_unprocessed_image(self, idx: int) -> torch.Tensor:
        """
        Return the resized RGB image as float32 tensor in [0,1], shape (3,H,W),
        WITHOUT ImageNet normalization. Safe for `.requires_grad_()`.
        """
        n = len(self.image_files)
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} is out of range for dataset of length {n}.")

        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img_resized = self.resize_img(img)
        img_t = self.to_tensor(img_resized)        # [0,1] float32 (3,H,W)
        return img_t.to(torch.float32, copy=False)

    def get_unprocessed_sample(self, idx: int):
        """
        Returns (raw_image_float32_[0..1], mask_long_{0,1}) after resizing.
        Image is NOT normalized.
        """
        x = self.get_unprocessed_image(idx)
        m = self._mask_to_long01(
            self.resize_mask(Image.open(self.mask_files[idx]).convert("L"))
        )
        return x, m

    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        Kept for compatibility; returns uint8 for visualization.
        Not used by CRP .requires_grad_() paths.
        """
        if data.dtype.is_floating_point:
            data = (data.clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            data = data.to(torch.uint8)
        return data.detach().cpu()

    @staticmethod
    def _mask_to_long01(mask_pil: Image.Image) -> torch.Tensor:
        m = transforms.functional.pil_to_tensor(mask_pil).squeeze(0)  # uint8 (H,W)
        uniq = torch.unique(m)
        if set(uniq.tolist()) <= {0, 255}:
            m = (m > 127).to(torch.uint8)
        elif set(uniq.tolist()) <= {0, 1}:
            m = m.to(torch.uint8)
        else:
            m = (m > 0).to(torch.uint8)
        return m.to(torch.long)
