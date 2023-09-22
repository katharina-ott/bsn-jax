from dataclasses import dataclass
from enum import Enum
from typing import List, Type, Union

from data import IntegrationDataset, GenzContinuousDataSet1D, GenzDiscontinuousDataSet1D, GenzGaussianDataSet1D, \
    GenzCornerpeakDataSet1D, GenzOscillatoryDataSet1D, GenzProductpeakDataSet1D


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


