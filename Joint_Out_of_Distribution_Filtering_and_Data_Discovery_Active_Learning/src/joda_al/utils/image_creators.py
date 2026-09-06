import colorsys
from typing import List, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from joda_al.data_loaders.sampler import SubsetSequentialSampler


def create_t_sne_plot(train_dataset, labeled_idx, unlabeled_idx, selected_idx, models, training_config, device, handler=None,ds_handler=None, plot_ind_ood=False):
    unlabeled_not_selected_idx = [idx for idx in unlabeled_idx if idx not in selected_idx]
    if ds_handler is not None:
        collate_fn=ds_handler.fn_collate
    else:
        collate_fn=None
    training_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"],
                                 sampler=SubsetSequentialSampler(
                                     labeled_idx + unlabeled_not_selected_idx + selected_idx),
                                 collate_fn=collate_fn,
                                 num_workers=training_config["num_workers"])


    features = handler.get_features(models, training_loader, device)

    tsne = TSNE(n_components=2, learning_rate='auto')
    tsne_results = tsne.fit_transform(features.cpu().numpy())

    fig = plt.figure()
    plt.scatter(tsne_results[:len(labeled_idx), 0], tsne_results[:len(labeled_idx), 1], alpha=0.6, label="Labeled", s=10)
    plt.scatter(tsne_results[len(labeled_idx):len(labeled_idx) + len(unlabeled_not_selected_idx), 0],
                tsne_results[len(labeled_idx):len(labeled_idx) + len(unlabeled_not_selected_idx), 1], marker="D",
                alpha=0.6, label="Unlabeled", s=10)
    plt.scatter(tsne_results[len(labeled_idx) + len(unlabeled_not_selected_idx):, 0],
                tsne_results[len(labeled_idx) + len(unlabeled_not_selected_idx):, 1], marker="x", alpha=0.6,
                label="Selected", s=10)
    plt.legend()
    # plt.title("t-SNE analysis - perplexity 30")
    plt.close(fig)
    if plot_ind_ood:
        fig2 = create_ind_ood_plot(labeled_idx, unlabeled_not_selected_idx, selected_idx, train_dataset, tsne_results)
        return fig, fig2
    return fig, None


def create_ind_ood_plot(labeled_idx, unlabeled_not_selected_idx, selected_idx, train_dataset, tsne_results):
        fig2 = plt.figure()
        idcs_ind = [i for i, idx in enumerate(labeled_idx + unlabeled_not_selected_idx + selected_idx) if
                    train_dataset[idx][1] >= 0 and train_dataset[idx][1] < train_dataset.num_classes]
        idcs_nearood = [i for i, idx in enumerate(labeled_idx + unlabeled_not_selected_idx + selected_idx) if
                        train_dataset[idx][1] >= train_dataset.num_classes]
        idcs_farood = [i for i, idx in enumerate(labeled_idx + unlabeled_not_selected_idx + selected_idx) if
                       train_dataset[idx][1] < 0]
        plt.scatter(tsne_results[idcs_ind, 0], tsne_results[idcs_ind, 1], alpha=0.6, label="inD", s=10)
        plt.scatter(tsne_results[idcs_nearood, 0], tsne_results[idcs_nearood, 1], alpha=0.6, label="nearOOD", s=10)
        if len(idcs_farood) > 0:
            plt.scatter(tsne_results[idcs_farood, 0], tsne_results[idcs_farood, 1], alpha=0.6, label="farOOD", s=10)


        plt.legend()
        plt.close(fig2)
        return fig2


def create_inference_image(prediction, dataset_config):
    colored_prediction = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    for boundary_type in range(dataset_config["num_classes"]):
        rgb_color = dataset_config["color_list"][boundary_type]
        mask = np.ma.masked_where(
            prediction[:, :, boundary_type] > 0.5,
            prediction[:, :, boundary_type],
        )
        colored_prediction[mask.mask, :] = np.array(rgb_color)
    return colored_prediction


def create_label_image(prediction, dataset_config):
    colored_prediction = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    for boundary_type in range(dataset_config["num_classes"]):
        rgb_color = dataset_config["color_list"][boundary_type]
        mask = np.ma.masked_where(
            prediction[:, :] == boundary_type,
            prediction[:, :],
        )
        colored_prediction[mask.mask, :] = np.array(rgb_color)
    return colored_prediction


def create_prediction_image(input :np.ndarray, prediction, dataset_config):
    fig = plt.figure()

    input = np.transpose(input, (1, 2, 0))
    if len(prediction.shape) == 3:
        prediction = np.transpose(prediction, (1, 2, 0))
        inference_image = create_inference_image(prediction, dataset_config)
    elif len(prediction.shape) == 2:
        inference_image = create_label_image(prediction, dataset_config)
    else:
        raise NotImplementedError()
    plt.imshow(input)
    masked_inference = np.ma.masked_where(
        inference_image[:, :] == np.array([0, 0, 0]), inference_image
    )
    plt.imshow(masked_inference,alpha=0.5)
    return fig


def create_detection_image(input_sample, prediction:Dict, dataset_config):
    fig = plt.figure()

    input_sample = np.transpose(input_sample, (1, 2, 0))
    plt.imshow(input_sample)
    for boxes,label,score in zip(prediction["boxes"],prediction["labels"],prediction.get("scores",[1.0]*prediction["labels"].shape[0])):
        xmin, ymin, xmax, ymax = boxes
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        width= xmax-xmin
        height=ymax-ymin
        # rgb_color = dataset_config["color_list"][pred["scores"].argmax()]
        rect = Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor='none')
        plt.gca().add_patch(rect)

    return fig


def get_color_range(n: int):
    HSV_tuples = [(x * 1.0 / n, 1.0, 1.0) for x in range(n)]
    RGB_tuples = [torch.tensor(colorsys.hsv_to_rgb(*x)) for x in HSV_tuples]
    return RGB_tuples


def segmentation_to_img(segmentation: torch.Tensor, num_classes: int, colors=None):
    if colors is None:
        colors = get_color_range(num_classes)

    batches = segmentation.shape[0]

    img = torch.zeros((batches, 3, segmentation.shape[1], segmentation.shape[2]))
    for b in range(batches):
        for i in range(num_classes):
            img[b, :, segmentation[b] == i] = colors[i].reshape((3, 1))
    return img
