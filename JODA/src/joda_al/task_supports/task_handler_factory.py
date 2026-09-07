from joda_al.defintions import TaskDef
from joda_al.task_supports.pytorch.ClassificationHandler import ClassificationHandler
from joda_al.task_supports.pytorch.SegmentationHandler import SegementationHandler


def get_task_handler(task, engine=None):
    """
    Currently only pytorch is supported.
    """
    return get_pytorch_task_handler(task)


def get_pytorch_task_handler(task):
    if task == TaskDef.classification:
        return ClassificationHandler
    elif task == TaskDef.semanticSegmentation:
        return SegementationHandler
    else:
        raise NotImplementedError(f"Task {task} is not supported.")