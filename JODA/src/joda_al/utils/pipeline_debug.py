"""Opt-in diagnostics that observe real batches without extra data iteration."""

from collections import Counter

import torch


PREFIX = "[PIPELINE_DEBUG]"


def enabled(config):
    return bool(config.get("debug_pipeline", False))


def selected_epoch(config, epoch):
    epochs = config.get("debug_epochs", [0, 1, 59, 60, 119, 120, 159, 160, 199])
    return epoch is None or int(epoch) in {int(value) for value in epochs}


def emit(message):
    print(f"{PREFIX} {message}", flush=True)


def tensor_summary(value):
    if not torch.is_tensor(value):
        return {"type": type(value).__name__}
    value = value.detach()
    result = {"shape": tuple(value.shape), "dtype": str(value.dtype),
              "device": str(value.device), "numel": value.numel()}
    if value.numel() and (value.is_floating_point() or value.is_complex()):
        finite = torch.isfinite(value)
        result.update(finite=int(finite.sum()), nonfinite=int((~finite).sum()))
        if finite.any():
            numbers = value[finite].float()
            result.update(min=float(numbers.min()), max=float(numbers.max()),
                          mean=float(numbers.mean()), std=float(numbers.std(unbiased=False)))
        if value.ndim == 4 and value.shape[1] in (1, 3):
            channels = value.float().permute(1, 0, 2, 3).reshape(value.shape[1], -1)
            result["channel_mean"] = channels.mean(1).cpu().tolist()
            result["channel_std"] = channels.std(1, unbiased=False).cpu().tolist()
            result["channel_min"] = channels.min(1).values.cpu().tolist()
            result["channel_max"] = channels.max(1).values.cpu().tolist()
    return result


def label_summary(labels, num_classes=None):
    if not torch.is_tensor(labels):
        return {"type": type(labels).__name__}
    flat = labels.detach().cpu().long().reshape(-1)
    counts = Counter(int(value) for value in flat.tolist())
    result = {"shape": tuple(labels.shape), "dtype": str(labels.dtype),
              "min": int(flat.min()) if flat.numel() else None,
              "max": int(flat.max()) if flat.numel() else None,
              "unique": len(counts), "histogram": dict(sorted(counts.items()))}
    if num_classes is not None:
        result.update(negative=int((flat < 0).sum()),
                      current_ind=int(((flat >= 0) & (flat < num_classes)).sum()),
                      not_current_ind=int((flat >= num_classes).sum()))
    return result


def register_leaf_hooks(model, context, maximum=200):
    handles = []
    counter = {"value": 0}

    def make_hook(name):
        def hook(_module, inputs, output):
            if counter["value"] >= maximum:
                return
            counter["value"] += 1
            input_value = inputs[0] if isinstance(inputs, tuple) and inputs else inputs
            output_value = output[0] if isinstance(output, (tuple, list)) and output else output
            emit(f"{context} module={name} input={tensor_summary(input_value)} "
                 f"output={tensor_summary(output_value)}")
        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def optimizer_summary(optimizers):
    return {name: [{"lr": float(group["lr"]),
                    "momentum": group.get("momentum"),
                    "weight_decay": group.get("weight_decay"),
                    "nesterov": group.get("nesterov"),
                    "parameters": sum(p.numel() for p in group["params"])}
                   for group in optimizer.param_groups]
            for name, optimizer in optimizers.items()}


def dataset_chain_summary(dataset, maximum_depth=8):
    """Describe nested Subset/AlteredDataset wrappers without fetching images."""
    chain = []
    current = dataset
    for _ in range(maximum_depth):
        item = {"type": f"{type(current).__module__}.{type(current).__name__}",
                "length": len(current)}
        if hasattr(current, "indices"):
            indices = list(current.indices)
            item["indices"] = {"count": len(indices), "unique": len(set(indices)),
                               "first": indices[:5], "last": indices[-5:]}
        if hasattr(current, "transform"):
            item["transform"] = repr(current.transform)
        if hasattr(current, "transforms"):
            item["transforms"] = repr(current.transforms)
        if hasattr(current, "map") and isinstance(current.map, dict):
            item["label_map"] = dict(current.map)
        chain.append(item)
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None or next_dataset is current:
            break
        current = next_dataset
    return chain


def model_norms(model):
    parameter_sq = gradient_sq = 0.0
    bad_parameters = bad_gradients = 0
    for parameter in model.parameters():
        value = parameter.detach()
        parameter_sq += float(value.float().pow(2).sum())
        bad_parameters += int((~torch.isfinite(value)).sum())
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            gradient_sq += float(gradient.float().pow(2).sum())
            bad_gradients += int((~torch.isfinite(gradient)).sum())
    return {"parameter_l2": parameter_sq ** 0.5, "gradient_l2": gradient_sq ** 0.5,
            "nonfinite_parameters": bad_parameters, "nonfinite_gradients": bad_gradients}
