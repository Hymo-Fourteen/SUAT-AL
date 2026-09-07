from typing import Dict, List, Union, Optional
from joda_al.defintions import TaskDef
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.logging_utils.training_logger import append_csv_row, global_write_scalar


def write_csv_file_by_task(
    experiment_path: str,
    task: TaskDef,
    loss: List[float],
    test_metric: Dict,
    val_metric: List[Dict],
    value_unlabeled_data: Optional[List] = None,
    selected_unlabeled_idx: Optional[List[int]] = None,
) -> None:
    """
    Write CSV file by task.

    Args:
        experiment_path (str): Path to the experiment directory.
        task (TaskDef): Task definition.
        loss (float): Training loss.
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics. during training
        value_unlabeled_data (Optional[List], optional): Valued unlabeled data. Defaults to None.
        selected_unlabeled_idx (Optional[List[int]], optional): Selected unlabeled indices. Defaults to None.
    """
    if value_unlabeled_data is None:
        value_unlabeled_data = []
    if selected_unlabeled_idx is None:
        selected_unlabeled_idx = []

    stats = {"loss": loss}
    if task in {TaskDef.classification, TaskDef.zeroclassification}:
        stats.update(csv_file_cls(test_metric, val_metric))
    elif task == TaskDef.semanticSegmentation:
        stats.update(csv_file_semseg(test_metric, val_metric))
    stats.update(
        {
            "valued_unlabeled_data": list(value_unlabeled_data),
            "selected_idx": selected_unlabeled_idx,
        }
    )
    append_csv_row(experiment_path, "stats_file", stats)


def csv_file_cls(test_metric: Dict, val_metric: List[Dict]) -> Dict:
    """
    Prepare classification metrics for CSV.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.

    Returns:
        Dict: Processed metrics for CSV.
    """
    metrics = {
        "val_acc": [metric["acc"] for metric in val_metric],
        "test_acc": test_metric["acc"],
        "test_f1": test_metric["f1"],
    }
    if "avg_acc" in test_metric.keys():
        metrics["test_avg_acc"] = test_metric["avg_acc"]
    return metrics


def csv_file_semseg(test_metric: Dict, val_metric: List[Dict]) -> Dict:
    """
    Prepare semantic segmentation metrics for CSV.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.

    Returns:
        Dict: Processed metrics for CSV.
    """
    return {
        "val_mIoU": [metric["mIoU"] for metric in val_metric],
        "test_mIoU": test_metric["mIoU"],
        "val_MIoU": [metric["MIoU"] for metric in val_metric],
        "test_MIoU": test_metric["MIoU"],
        "val_cmIoU": [metric["cmIoU"] for metric in val_metric],
        "test_cmIoU": test_metric["cmIoU"],
        "val_catIoU": [metric.get("catIoU",0) for metric in val_metric],
        "test_catIoU": test_metric.get("catIoU",0),
    }


def csv_file_2d_detection(test_metric: Dict, val_metric: List[Dict]) -> Dict:
    """
    Prepare 2D detection metrics for CSV.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.

    Returns:
        Dict: Processed metrics for CSV.
    """
    return {
        "val_mAP": [metric["mAP"] for metric in val_metric],
        "test_mAP": test_metric["mAP"],
    }


def write_cycle_stats_by_task(
    task: TaskDef, loss: float, val_metric: List[Dict], test_metric: Dict, cycle_num: int, only_test: bool = False
) -> None:
    """
    Write cycle statistics by task.

    Args:
        task (TaskDef): Task definition.
        loss (float): Training loss.
        val_metric (List[Dict]): Validation metrics.
        test_metric (Dict): Test metrics.
        cycle_num (int): Cycle number.
        only_test (bool, optional): Flag to indicate if only test metrics should be written. Defaults to False.
    """
    global_write_scalar("Loss/Cycle/train", min(loss), cycle_num)
    if task in {TaskDef.classification, TaskDef.zeroclassification}:
        write_cycle_cls_stats(test_metric, val_metric, cycle_num, only_test)
    elif task == TaskDef.semanticSegmentation:
        write_cycle_semseg_stats(test_metric, val_metric, cycle_num)


def write_cycle_cls_stats(test_metric: Dict, val_metric: List[Dict], cycle_num: int, only_test: bool = False) -> None:
    """
    Write classification cycle statistics.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.
        cycle_num (int): Cycle number.
        only_test (bool, optional): Flag to indicate if only test metrics should be written. Defaults to False.
    """
    gl_info(f"Test Acc {test_metric['acc']}")
    gl_info(f"Test F1 {test_metric['f1']}")
    if "avg_acc" in test_metric.keys():
        gl_info(f"Test Avg Acc {test_metric['avg_acc']}")
    if "avg_acc" in val_metric[0].keys():
        gl_info(f"Val Avg Acc {max([metric['avg_acc'] for metric in val_metric])}")
    if not only_test:
        global_write_scalar(
            "Acc/Cycle/val", max([metric["acc"] for metric in val_metric]), cycle_num
        )
        global_write_scalar(
            "F1/Cycle/val", max([metric["acc"] for metric in val_metric]), cycle_num
        )
        if "avg_acc" in val_metric[0].keys():
            global_write_scalar(
                "AvgAcc/Cycle/val",
                max([metric["avg_acc"] for metric in val_metric]),
                cycle_num,
            )
    global_write_scalar("Acc/Cycle/test", test_metric["acc"], cycle_num)
    for i in range(3):
        if test_metric.get(f"acc_{i}", None) is not None:
            global_write_scalar(f"Acc{i}/Cycle/test", test_metric["avg_acc"], cycle_num)
    global_write_scalar("F1/Cycle/test", test_metric["f1"], cycle_num)
    if "avg_acc" in test_metric.keys():
        global_write_scalar("AvgAcc/Cycle/test", test_metric["avg_acc"], cycle_num)
        for i in range(3):
            if test_metric.get(f"avg_acc_{i}",None) is not None:
                global_write_scalar(f"AvgAcc{i}/Cycle/test", test_metric[f"avg_acc_{i}"], cycle_num)



def write_cycle_semseg_stats(test_metric: Dict, val_metric: List[Dict], cycle_num: int) -> None:
    """
    Write semantic segmentation cycle statistics.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.
        cycle_num (int): Cycle number.
    """
    gl_info(f"Test mIoU {test_metric['mIoU']}")
    global_write_scalar(
        "mIoU/Cycle/val", max([metric["mIoU"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("mIoU/Cycle/test", test_metric["mIoU"], cycle_num)
    global_write_scalar(
        "MIoU/Cycle/val", max([metric["MIoU"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("MIoU/Cycle/test", test_metric["MIoU"], cycle_num)
    global_write_scalar(
        "cmIoU/Cycle/val", max([metric["cmIoU"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("cmIoU/Cycle/test", test_metric["cmIoU"], cycle_num)


def write_cycle_detection_stats(test_metric: Dict, val_metric: List[Dict], cycle_num: int) -> None:
    """
    Write 2D detection cycle statistics.

    Args:
        test_metric (Dict): Test metrics.
        val_metric (List[Dict]): Validation metrics.
        cycle_num (int): Cycle number.
    """
    gl_info(f"Test mAP {test_metric['mAP']}")
    global_write_scalar(
        "mAP/Cycle/val", max([metric["mAP"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("mAP/Cycle/test", test_metric["mAP"], cycle_num)

    global_write_scalar(
        "mAP50/Cycle/val", max([metric["mAP50"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("mAP50/Cycle/test", test_metric["mAP50"], cycle_num)

    global_write_scalar(
        "mAP75/Cycle/val", max([metric["mAP75"] for metric in val_metric]), cycle_num
    )
    global_write_scalar("mAP75/Cycle/test", test_metric["mAP75"], cycle_num)
