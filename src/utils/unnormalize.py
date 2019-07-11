"""Unnormalize.

Reverse the normalize transform so that viewed images make sense.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    "Simple way to inverse transform ? Normalization"
    <https://discuss.pytorch.org/t/simple-way-to-inverse-transform-
    normalization/4821/2>
"""


class UnNormalize:
    def __init__(self, mean, std):
        """Reverse normalization done by torchvision.transforms.normalize."""
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
        Returns:
            Tensor: Unnormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)

        return tensor
