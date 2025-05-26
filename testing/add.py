import torch

class AddChannel:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = img.unsqueeze(0)
            elif img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Invalid image shape: {img.shape}")
        else:
            raise ValueError(f"Invalid image type: {type(img)}")
        return img


