from dataclasses import dataclass
from enum import Enum
from typing import List, Type, Union

from bsn_jax.data.data import IntegrationDataset


class MethodEnum(str, Enum):
    """
    List of possible optimizers,
    adam uses the `JAX` implementation, the other optimizers use the `scipy` implementation.
    """

    ADAM = "adam"
    CG = "CG"
    BFGS = "BFGS"
    NEWTON_CG = "Newton-CG"
    L_BFGS_B = "L-BFGS-B"


@dataclass
class Options:
    step_size: float
    num_epochs: int
    layer_sizes: List[List[int]]
    n: int
    data_class: Type[IntegrationDataset]
    method: Union[MethodEnum, str]
