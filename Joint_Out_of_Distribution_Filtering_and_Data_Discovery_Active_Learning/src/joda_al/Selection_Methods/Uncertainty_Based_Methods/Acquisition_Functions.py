import numpy as np
import pandas as pd

import torch
from scipy.special import softmax
from scipy.stats import entropy
import scipy
from time import time
from itertools import combinations


def variance(results):
    """
    calculates the variance for the different predictions per class
    :param results: [num of predictions x classes]
    :return: variance
    """
    if len(results.shape) == 3 and results.shape[0] > 1:
        var = np.var(results, axis=1)
        return var
    var = np.var(results, axis=0)
    return var


def mean_variance(results):
    """
    calculates the variance for the different predictions meaned over the classes
    :param results: [num of predictions x classes]
    :return: mean of variance
    """
    var = variance(results)
    return np.mean(var)


def entropy(results):
    """
    This functions calculates the entropy, defined for the uncertainty case,
    see Yarin Gal
    :param results: Matrix of results [num of predictions x classes]
    :return: the sum of the entropy
    """
    if len(results.shape) == 3 and results.shape[0] > 1:
        mean = np.mean(results, axis=1)
        log_ent = np.log(mean)
        log_ent[log_ent == -np.inf] = 0
        entropy = -mean * log_ent
        return np.sum(entropy, axis=2)
    mean = np.mean(results, axis=0)
    log_ent = np.log(mean)
    log_ent[log_ent == -np.inf] = 0
    entropy = -mean * log_ent
    return np.sum(entropy)


def mutual_information(results):
    """
    This functions calculates the mutual information, defined for the uncertainty case,
    see Yarin Gal
    :param results: Matrix of results [num of predictions x classes]
    :return: mutual information
    """
    ent = entropy(results)
    log_res = np.log(results)
    log_res[log_res == -np.inf] = 0
    cross_ent = results * log_res
    cross_ent = np.mean(np.sum(cross_ent, axis=1))
    mi = ent + cross_ent
    return mi


def variation_ratio(results):
    """
    This functions calculates the variational ratio, defined for the uncertainty case,
    see Yarin Gal
    :param results: Matrix of results [num of predictions x classes]
    :return: variational ratio
    """
    classes = np.zeros(44)
    for r in results:
        classes[np.argmax(r)] += 1
    maxclass = np.max(classes)
    return 1 - maxclass / float(len(results))


def estimate_uncertainty(result, crit):
    """
    This function is a wrapper for all uncertainty criterions.
    :param result: output of the neuronal network [num_samples x num of predictions x classes]
    :param crit: select the uncertainty criterion
    :return: uncterainty of the samples
    """
    uncertainty = []
    num_data=result.shape[0]
    if crit == 'mi':
        for i in range(num_data):
            uncertainty.append(mutual_information(result[i, :, :]))
    if crit == 'ent':
        for i in range(num_data):
            uncertainty.append(entropy(result[i, :, :]))
    if crit == 'vr':
        for i in range(num_data):
            uncertainty.append(variation_ratio(result[i, :, :]))
    return uncertainty

## Part for 2D Detection


def calaculate_uncertainty_roi(output):
    for o in output:
        softmax_scores=o['pred_scores'].data.cpu().numpy()

    return


def localisation_tightness_all(rois_sampling, boxes_sampling, scores_sampling):
    """
    calculates the localisation tightness for all images and MC sets at once using matrix transformation and GPU
    computations
    :param rois_sampling: [samples x MC x roi x 4]
    :param boxes_sampling: [samples x MC x roi x classes x 4]
    :param scores_sampling: [samples x MC x roi x classes]
    :return:
    """
    #choose the bounding box with the highest softmax score
    maxs = scores_sampling.argmax(axis=3)
    dim1, dim2, dim3 = np.indices(maxs.shape)
    box_max = boxes_sampling[dim1, dim2, dim3, maxs]
    # Mean MC outputs

    #box_mean=box_max.swapaxis()
    #rois_sampling=rois_sampling.mean(axis=1)
    #Calculate IoU on GPU using tensorflow

    loc_tightness = tf_iou_flat(rois_sampling.reshape(-1, 4), box_max.reshape(-1, 4)).reshape(boxes_sampling.shape[0], boxes_sampling.shape[1], boxes_sampling.shape[2])
    #print(loc_tightness.shape)
    loc_tightness = np.nan_to_num(loc_tightness)
    # 1- x as IuO is between 0 and 1 -> 1 would be certain , 0 uncertain just fo switching
    return 1 - loc_tightness.mean(axis=1)


def localisation_tightness(rois_sampling, boxes_sampling, scores_sampling):
    """
    depricated!!!
    calculates the localisation tightness on the CPU... depricated
    :param rois_sampling: [samples x MC x roi x 4]
    :param boxes_sampling: [samples x MC x roi x classes x 4]
    :param scores_sampling: [samples x MC x roi x classes]
    :return:
    """
    loc_tightness = np.zeros((rois_sampling.shape[0], rois_sampling[0].shape[0]))
    box_max=np.zeros((rois_sampling[0].shape[0], 4))
    scores_sampling = np.swapaxes(scores_sampling, 0, 1)

    #print(rois_sampling.shape)

    for i, (rois, boxes, scores) in enumerate(zip(rois_sampling, boxes_sampling, scores_sampling)):
         #for j, (roi, box, score) in enumerate(zip(rois,boxes,scores)):
         #box_max=boxes[:,scores.argmax(axis=1),:]

        for j in range(rois_sampling[0].shape[0]):
            box_max[j] = boxes[i, scores.argmax(axis=1)[i], :]
        #print(rois.shape,boxes.shape,box_max.shape,loc_tightness.shape)
        #print(tf_iou_flat(rois.reshape(-1,4), box_max.reshape(-1,4)))
        loc_tightness[i] = tf_iou_flat(rois.reshape(-1, 4), box_max.reshape(-1, 4)).reshape(rois_sampling[0].shape[0])

        #loc_tightness[i] = tf_iou(rois.reshape(-1, 4), box_max.reshape(-1, 4)).diagonal().reshape(rois_sampling[0].shape[0])
    return 1 - loc_tightness.mean(axis=0) # 1- x as IuO is between 0 and 1 -> 1 would be certain , 0 uncertain just fo switching


def total_variance(rois_sampling, boxes_sampling, scores_sampling):
    """

    :param rois_sampling:
    :param boxes_sampling:
    :param scores_sampling:
    :return:
    """
    l=boxes_sampling.shape[1]
    maxs = scores_sampling.argmax(axis=3)
    dim1, dim2, dim3 = np.indices(maxs.shape)
    box_max = boxes_sampling[dim1, dim2, dim3, maxs] #[samples x MC x roi x 4]

    return

# Part for clustering


def consensus_uncertainty(rois_sampling, boxes_sampling, scores_sampling, mode="vr"):
    """
    calculates the Consensus Score for all images and MC sets at once using matrix transformation and GPU
    computations
    :param rois_sampling: [samples x MC x roi x 4]
    :param boxes_sampling: [samples x MC x roi x classes x 4]
    :param scores_sampling: [samples x MC x roi x classes]
    :return:
    """
    l = boxes_sampling.shape[1]
    #combination = combinations([i for i in range(l)], 2)
    combination = [(0, i+1)for i in range(l-1)]
    classmatch=np.zeros_like(scores_sampling)
    #print(box_max.shape, classmatch.shape)
    #max class in the roi problem
    classmatch[:, 0, :, :] = scores_sampling[:, 0, :, :]
    scores = np.zeros((classmatch.shape[0], classmatch.shape[1]-1, classmatch.shape[2]))
    for i, comb in enumerate(combination):
        score = consensus_score_2BB(rois_sampling[:, comb[0], :, :], rois_sampling[:, comb[1], :, :]) #[samples x Roi x Roi] At first only the mayor class per roi
        bbmatch = score.argmax(axis=1)
        scores[:, i] = score.max(axis=1)
        dim1, dim2 = np.indices(bbmatch.shape)
        class_max = scores_sampling[:, comb[1], :, :]
        #print(maxs.shape, class_max.shape,classmatch.shape)
        classmatch[:, comb[1], :, :] = class_max[dim1, bbmatch[dim1, dim2], :]

    uncertainty = [estimate_uncertainty(classmatch[i], mode) for i in range(classmatch.shape[0])] #[samples x MC x roi x classes]
    return np.array(uncertainty)


def consensus_score(rois_sampling, boxes_sampling, scores_sampling, mode="lvr"):
    """
    calculates the Consensus Score for all images and MC sets at once using matrix transformation and GPU
    computations
    :param rois_sampling: [samples x MC x roi x 4]
    :param boxes_sampling: [samples x MC x roi x classes x 4]
    :param scores_sampling: [samples x MC x roi x classes]
    :return:
    """
    l = boxes_sampling.shape[1]
    maxs = scores_sampling.argmax(axis=3)
    dim1, dim2, dim3 = np.indices(maxs.shape)
    box_max = boxes_sampling[dim1, dim2, dim3, maxs] # [samples x MC x roi x 4]

    #combination = combinations([i for i in range(l)], 2)
    combination =[(0, i+1)for i in range(l-1)]
    #print(combination)
    classmatch=np.zeros_like(maxs)
    #print(box_max.shape, classmatch.shape)
    #max class in the roi problem
    classmatch[:, 0, :] = maxs[:, 0, :]
    scores=np.zeros((classmatch.shape[0], classmatch.shape[1]-1, classmatch.shape[2]))
    for i,comb in enumerate(combination):
        score=consensus_score_2BB(box_max[:, comb[0], :, :], box_max[:, comb[1], :, :]) # [samples x Roi x Roi] At first only the mayor class per roi
        bbmatch = score.argmax(axis=1)
        scores[:, i] = score.max(axis=1)
        dim1, dim2 = np.indices(bbmatch.shape)
        class_max = maxs[:, comb[1], :]
        #print(maxs.shape, class_max.shape,classmatch.shape)
        classmatch[:, comb[1], :] = class_max[dim1, bbmatch[dim1, dim2]]
    _, count = scipy.stats.mode(classmatch, axis=1)
    #print(count.shape)
    count = count/maxs.shape[1]
    count = count.reshape(count.shape[0], count.shape[2])
    # consensus or consensus vr
    con_score = scores.min(axis=1)
    if mode == "lvr":
        con_score = np.multiply(con_score, count)
    con_score_sum = con_score.sum(axis=1).reshape(-1, 1)
    return con_score_sum, count


def consensus_score_2BB_tf(boxes1, boxes2):
    """
    Tensorflow function which calculates the consensus score for two set of bounding boxes
    :param boxes1: set 1
    :param boxes2: set 2
    :return:
    """
    import tensorflow as tf
    boxes1 = np.swapaxes(boxes1, 0, 1)
    boxes2 = np.swapaxes(boxes2, 0, 1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    iou_np=np.zeros((boxes1.shape[0], boxes1.shape[1], boxes1.shape[0]))
    length=boxes1.shape[1]
    tousands=length/1000

    tf_bboxes1 = tf.placeholder(dtype=tf.float16, shape=[None, None, 4])
    tf_bboxes2 = tf.placeholder(dtype=tf.float16, shape=[None, None, 4])

    def run(tb1, tb2):
        y11, x11, y12, x12 = tf.split(tb1, 4, axis=2)
        y21, x21, y22, x22 = tf.split(tb2, 4, axis=2)

        xA = tf.maximum(x11, tf.transpose(x21))
        yA = tf.maximum(y11, tf.transpose(y21))
        xB = tf.minimum(x12, tf.transpose(x22))
        yB = tf.minimum(y12, tf.transpose(y22))

        interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

        boxAArea = tf.abs((x12 - x11) * (y12 - y11))
        boxBArea = tf.abs((x22 - x21) * (y22 - y21))
        # boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        # boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

        return iou, y11, xA, interArea

    op = run(tf_bboxes1, tf_bboxes2)

    # sess.run(op, feed_dict={tf_bboxes1: boxes1, tf_bboxes2: boxes2})
    tic = time()

    #iou, y11, xA, interArea = sess.run(op, feed_dict={tf_bboxes1: boxes1, tf_bboxes2: boxes2})
    it =0
    while it<tousands:
        it_t=it*1000
        iou_np[:, it_t:it_t+1000, :], _, _, _ = sess.run(op, feed_dict={tf_bboxes1: boxes1[:,it_t:it_t+1000,:], tf_bboxes2: boxes2[:,it_t:it_t+1000,:]})
        print(it)
        it+=1
    it_t = it * 1000
    iou_np[:,it_t:,:], _, _ , _ = sess.run(op, feed_dict={tf_bboxes1: boxes1[:,it_t:,:], tf_bboxes2: boxes2[:,it_t:,:]})
    #nan intr
    iou = np.nan_to_num(iou_np)
    sess.close()
    tf.reset_default_graph()
    toc = time()
    iou = np.swapaxes(iou, 0, 1)
    print(toc - tic)
    return iou # ,toc - ticdef tf_iou(boxes1, boxes2):


def consensus_score_2BB(boxes1, boxes2):
    """
    PyTorch function which calculates the consensus score for two sets of bounding boxes
    :param boxes1: set 1
    :param boxes2: set 2
    :return:
    """
    boxes1 = np.swapaxes(boxes1, 0, 1)
    boxes2 = np.swapaxes(boxes2, 0, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iou_np = np.zeros((boxes1.shape[0], boxes1.shape[1], boxes1.shape[0]))
    length = boxes1.shape[1]
    thousands = length // 1000

    def run(tb1, tb2):
        y11, x11, y12, x12 = torch.split(tb1, 1, dim=2)
        y21, x21, y22, x22 = torch.split(tb2, 1, dim=2)

        xA = torch.max(x11, x21.transpose(0, 2))
        yA = torch.max(y11, y21.transpose(0, 2))
        xB = torch.min(x12, x22.transpose(0, 2))
        yB = torch.min(y12, y22.transpose(0, 2))

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        boxAArea = torch.abs((x12 - x11) * (y12 - y11))
        boxBArea = torch.abs((x22 - x21) * (y22 - y21))

        iou = interArea / (boxAArea + boxBArea.transpose(0, 2) - interArea)

        return iou, y11, xA, interArea

    tic = time()

    boxes1 = torch.tensor(boxes1, dtype=torch.float32, device=device)
    boxes2 = torch.tensor(boxes2, dtype=torch.float32, device=device)

    iou_np = np.zeros((boxes1.shape[0], boxes1.shape[1], boxes1.shape[0]))

    it = 0
    while it < thousands:
        it_t = it * 1000
        iou_np[:, it_t:it_t + 1000, :], _, _, _ = run(boxes1[:, it_t:it_t + 1000, :], boxes2[:, it_t:it_t + 1000, :])
        print(it)
        it += 1
    it_t = it * 1000
    iou_np[:, it_t:, :], _, _, _ = run(boxes1[:, it_t:, :], boxes2[:, it_t:, :])
    # nan intr
    iou = np.nan_to_num(iou_np)
    toc = time()
    iou = np.swapaxes(iou, 0, 1)
    print(toc - tic)
    return iou


def intersection_over_union(boxA, boxB):
    """
    Intersection over union, CPU version
    :param boxA:
    :param boxB:
    :return: IoU
    """
    # determine the (x, y)-coordinates of the intersection rectangle#from Model.faster_rcnn_dropout import FasterRCNN_Dropout
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def tf_iou(boxes1, boxes2):
    """
    Intersection over union on the GPU
    :param boxes1:
    :param boxes2:
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)

    tf_bboxes1 = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    tf_bboxes2 = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    def run(tb1, tb2):
        y11, x11, y12, x12 = tf.split(tb1, 4, axis=1)
        y21, x21, y22, x22 = tf.split(tb2, 4, axis=1)

        xA = tf.maximum(x11, tf.transpose(x21))
        yA = tf.maximum(y11, tf.transpose(y21))
        xB = tf.minimum(x12, tf.transpose(x22))
        yB = tf.minimum(y12, tf.transpose(y22))

        interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

        boxAArea = tf.abs((x12 - x11) * (y12 - y11))
        boxBArea = tf.abs((x22 - x21) * (y22 - y21))
        #boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        #boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

        return iou, y11, xA, interArea

    op = run(tf_bboxes1, tf_bboxes2)

    #sess.run(op, feed_dict={tf_bboxes1: boxes1, tf_bboxes2: boxes2})
    tic = time()
    iou, y11, xA, interArea=sess.run(op, feed_dict={tf_bboxes1: boxes1, tf_bboxes2: boxes2})
    toc = time()
    print(toc-tic)
    return iou, y11, xA, interArea #,toc - ticdef tf_iou(boxes1, boxes2):


def tf_iou_flat(boxes1, boxes2):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)

    tf_bboxes1 = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    tf_bboxes2 = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    def run(tb1, tb2):
        y11, x11, y12, x12 = tf.split(tb1, 4, axis=1)
        y21, x21, y22, x22 = tf.split(tb2, 4, axis=1)

        xA = tf.maximum(x11, x21)
        yA = tf.maximum(y11, y21)
        xB = tf.minimum(x12, x22)
        yB = tf.minimum(y12, y22)

        interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

        boxAArea = tf.abs((x12 - x11) * (y12 - y11))
        boxBArea = tf.abs((x22 - x21) * (y22 - y21))

        iou = interArea / (boxAArea + boxBArea - interArea)
        if boxAArea == 0 or boxBArea == 0:
            iou=0

        return iou

    op = run(tf_bboxes1, tf_bboxes2)
    tic = time()
    iou=sess.run(op, feed_dict={tf_bboxes1: boxes1, tf_bboxes2: boxes2})
    toc = time()
    print(toc-tic)
    return iou


def tf_area(boxes):
    """
    Tensorflow area of a box calculation
    :param boxes:
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)


    tf_bboxes = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    def run(tb1):
        y11, x11, y12, x12 = tf.split(tb1, 4, axis=1)

        boxAArea = tf.abs((x12 - x11) * (y12 - y11))

        return boxAArea

    op = run(tf_bboxes)

    area=sess.run(op, feed_dict={tf_bboxes: boxes})
    return area


def estimate_uncertainty_BB(bounding_box_list,predictions_list,prediction_scores, num_elements=1):
    # uncertainty of the classifications
    list_net_metrics=[]
    for i in range(num_elements):
        d = {'predictions': predictions_list, 'Score': prediction_scores, 'Bounding Box': bounding_box_list}
        df = pd.DataFrame(data=d)

##############################
# Amother Idea was to cluster the bounding box of the final prediction and get so multiple prediction for one object.
# Based on this multiple predictions the uncertainty can be estimated.
# Unfortunately this approach was not further examined

def cluster_bb(iou,clusterIoU=0.05):
    boxes=iou.shape[0]
    idx_done=[]
    clusters=[]
    for i in range(boxes):
        if i in idx_done:
            continue
        elements=cluster_elements=np.flatnonzero(iou[i] > clusterIoU)
        if len(elements)==0:
            clusters.append(np.array([i]))
        else:
            clusters.append(elements)
            idx_done+=elements.tolist()
    return clusters


def find_bb_correlations(boxes1,boxes2,classes1=[],classes2=[]):
    iou=tf_iou(boxes1, boxes2)
    boxes=iou.shape[0]
    correlation=[]
    for i in range(boxes):
        if classes1==[] and classes2==[]:
            correlation.append([i,np.argmax(iou[i]),np.max(iou[i])])
        else:
            correlation.append([i,np.argmax(iou[i]),np.max(iou[i]),classes1[i],classes2[np.argmax(iou[i])]])
    return correlation


def metrics(coordinate_set, metric='var'):
    if metric=='var':
        return np.var(coordinate_set)
    elif metric=='ent':
        return scipy.stats.entropy(coordinate_set)


def bounding_box_metrics(clusters, boxes):
    centers=[]
    metric=[]
    for cluster in clusters:
        center=np.array([boxes[cluster,2]-boxes[cluster,0],boxes[cluster,3]-boxes[cluster,1]])
        centers.append(np.array([boxes[cluster,2]-boxes[cluster,0],boxes[cluster,3]-boxes[cluster,1]]))
        metric.append(np.array([metrics(center),metrics(boxes[cluster,0]),metrics(boxes[cluster,1]),metrics(boxes[cluster,2]),metrics(boxes[cluster,3])]))
    return metric


def softmax_nonzero(array):
    soft = np.zeros_like(array)
    for i, x in enumerate(array):
        if x == 0:
            soft[i] = 0
        else:
            soft[i]=np.exp(x)/sum(np.exp(array[array>0]))
    return soft


def cluster_prototype(clusters,boxes,predictions=None,weight_type='std'):
    prototypes=[]
    for cluster in clusters:
        box=np.array(4)
        if weight_type=='std' or predictions==None:
            box=np.average(boxes[cluster],axis=0)
        if weight_type=='score':
            box=np.average(boxes[cluster],weights=predictions[cluster],axis=0)
        prototypes.append(box)
    return np.array(prototypes)


def cluster_mult_classification_metrics(clusters_list, boxes_list, classes_list, predictions_list, num_classes=44):
    distribution=[]
    prototypes=[]
    clusters_num=[]
    for i in range(len(clusters_list)):
        clusters_num.append(len(clusters_list[i]))
        distribution.append(cluster_classification_metrics(clusters_list[i], classes_list[i], predictions_list[i]))
        prototypes.append(cluster_prototype(clusters_list[i], boxes_list[i], predictions_list[i]))
    if np.array(clusters_num).mean() == clusters_num[0]:
        print("Same number of clusters")
    else:
        print("Different number of clusters")
    cluster_predictions=np.zeros((max(clusters_num),num_classes,len(clusters_list)))
    for j in range(len(clusters_list[0])):
        cluster_predictions[j,:,0]=distribution[0][j]
    for i in range(len(clusters_list)-1):
        cor=find_bb_correlations(prototypes[0],prototypes[i+1])
        iou_acc = lambda x: x[2] > 0.4
        for c in filter(iou_acc, cor):
            print(c[0])
        #for c in cor if c[1] > 0.3:
            cluster_predictions[c[1],:,i+1]=distribution[i+1][c[1]]
    uncertainty=estimate_uncertainty(cluster_predictions,'mi')
    return cluster_predictions, uncertainty
    #estimate_uncertainty(distribution, 'ent')


def cluster_classification_metrics(clusters, classes, predictions):
    distribution=[]
    for cluster in clusters:
        prob=np.zeros(44)
        for c in cluster:
            prob[classes[c]]+=predictions[c]
        #print(prob)
        #distribution.append(softmax(prob))
        #distribution.append(prob/np.max(prob))
        distribution.append(prob / np.sum(prob))
    return distribution


def estimate_classification_uncertainy(l_box, l_cla, l_sco):
    clusters=[]
    for i in range(len(l_box)):
        clusters.append(cluster_bb(tf_iou(l_box[i], l_box[i])))

    return cluster_mult_classification_metrics(clusters, l_box, l_cla, l_sco)

