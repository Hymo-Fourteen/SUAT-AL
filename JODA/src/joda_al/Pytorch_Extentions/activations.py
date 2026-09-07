import torch
from torch import Tensor
from torch.nn import Module


class PenalizedTanh(Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, penalty=0.5):
        super(PenalizedTanh, self).__init__()
        self.penalty = penalty

    def forward(self, input: Tensor) -> Tensor:
        zero_vec=torch.zeros_like(input)
        return torch.maximum(zero_vec, torch.tanh(input)) + torch.minimum(zero_vec, self.penalty * torch.tanh(input))
        # if input >= 0:
        #     return torch.tanh(input)
        # else:
        #     return self.penalty * torch.tanh(input)