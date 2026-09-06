from enum import Enum
from typing import TypeVar

from torch.utils.data import Dataset, Subset


class TaskDef(str, Enum):
    classification = "classification"
    semanticSegmentation = "semSeg"
    objectDetection = "object2Ddet"


class PreProcessingDef(str, Enum):
    norm = "norm"
    pretrained_norm = "preNorm"
    local_maxed = "local"
    equalized = "equalized"
    none = "none"


class ActiveLearningTrainingStrategies(str, Enum):
    retrain = "retrain"
    reuse = "reuse"
    continuous = "continuous"


class ActiveLearningScenario(str, Enum):
    stream = "Stream"
    pool = "Pool"
    streamBatch = "StreamBatch"
    poolStream = "PoolStream"
    multi_stream_pool = "MultiStreamPool"
    full = "Full"


class Scenario(str, Enum):
    unc = "Uncertainty"
    ood = "Out-of-Distribution"
    al = "Active-Learning"
    train = "train"
    eval = "eval"

class DataUpdateScenario(str, Enum):
    standard = "standard"
    osal = "osal" # Open Set Active Learning - Having OOD data in the pool
    osal_extending = "osal-extending" # Open Set Discovery Active Learning - Having OOD and Novel data in the pool
    osal_near_far = "osal-near-far" # Open Set Active Learning with near and far OOD (two types of OOD) data in the pool
    discovery_al = "discovery-al" # Discovery Active Learning - Having OOD data in the pool


class Engine(str, Enum):
    pytorch = "pytorch"


SubDataset = TypeVar("SubDataset", bound=Dataset)
SubSubset = TypeVar("SubSubset", bound=Subset)
