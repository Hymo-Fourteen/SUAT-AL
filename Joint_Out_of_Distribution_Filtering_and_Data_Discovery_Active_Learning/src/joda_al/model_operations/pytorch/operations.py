import os
import torch
from tqdm.auto import tqdm

from joda_al.models.model_lib import EnsembleModel
from joda_al.utils.logging_utils.settings import get_global_verbosity, get_global_track_weights
from joda_al.defintions import TaskDef
from joda_al.models.model_utils import save_model
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.method_utils import is_loss_learning, is_method_of
from joda_al.task_supports.task_handler_factory import get_task_handler
from joda_al.utils.logging_utils.training_logger import global_write_scalar, global_write_histogram

EARLY_STOPPING_METRICS = ["val_acc", "val_loss", "val_mIoU", "val_mAP", "val_MIoU"]


def train_epoch(models, method, task, criterion, optimizers, dataloader, training_config, device, epoch, epoch_loss):
    # train_step = get_train_step(task)
    if training_config["loss_func"] in ["OODCrossEntropy", "OpenCrossEntropy", "OutlierExposure", "EnergyExposure"]:
        train_step = get_task_handler(task).train_step_OpenSet
    else:
        train_step = get_task_handler(task).train_step

    # global iters
    mean_loss = []
    mean_module_loss = []
    num_data = len(dataloader)
    iteration=0

    # Warmup Scheduler
    warmup_lr_scheduler={}
    if epoch == 0 and training_config["warm_up"]:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)
        for opti_key, optimizer in optimizers.items():
            warmup_lr_scheduler[opti_key] = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

    ##### Just to reproduce paper
    if epoch == 1 and training_config["model"] == "ResNet110" and training_config["loss_func"] == "GorLoss":
        # for resnet110 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for _, optimizer in optimizers.items():
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 10
    ######

    with torch.autograd.set_detect_anomaly(True):
        with tqdm(dataloader, leave=False, total=num_data, desc='Iterations', unit="Batch", position=1, disable=get_global_verbosity()) as iterator:
            for data in iterator:
                iteration+=1
                #startt=time.time()
                with torch.cuda.device(device):

                    inputs = data[0].cuda()
                    if isinstance(data[1],dict):
                        labels = [{k: v.to(device) for k, v in data[1].items()}]
                    elif isinstance(data[1], list):
                        labels = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
                    else:
                        labels = data[1].cuda()

                # iters += 1 count global number of iterations
                #print("Pre",startt-time.time())
                for _, optimizer in optimizers.items():
                    if isinstance(optimizer,list):
                        for opti in optimizer:
                            opti.zero_grad()
                    else:
                        optimizer.zero_grad()

                if training_config["loss_func"] == "OODCrossEntropy":
                    loss, loss_module, _, _ = train_step(models, method, criterion, inputs, labels)
                    loss.backward()  # this is not a combined loss....
                    loss_module.backward()

                elif training_config["loss_func"] in ["OutlierExposure","EnergyExposure","OpenCrossEntropy"]:
                    loss, loss_module, _, _ = train_step(models, method, criterion, inputs, labels)

                    if isinstance(loss,list):
                        for l in loss:
                            l.backward()
                        loss=torch.mean(torch.stack(loss))

                    else:
                        loss.backward()
                else:
                    loss, loss_module, _ = train_step(models, method, criterion, inputs, labels, epoch, training_config, epoch_loss)
                    loss.backward() # this is a combined loss....


                for _, optimizer in optimizers.items():
                    if isinstance(optimizer,list):
                        for opti in optimizer:
                            opti.step()
                    else:
                        optimizer.step()
                for _, scheduler in warmup_lr_scheduler.items():
                    if isinstance(scheduler,list):
                        for opti in scheduler:
                            opti.step()
                    else:
                        scheduler.step()

            iterator.set_postfix_str(f"loss:{str(loss.detach().data.cpu().numpy())}")
            mean_loss.append(loss.detach().data.cpu().numpy())
            #if is_loss_learning(method):

            if loss_module is None:
                print("Check if 1 samples should be compared for loss")
            else:
                mean_module_loss.append(loss_module.detach().data.cpu().numpy())
                global_write_scalar('LossMod/Iteration/train', loss_module.detach().data.cpu().numpy(), num_data*epoch+iteration*training_config["batch_size"])

            global_write_scalar('Loss/Iteration/train', loss.detach().data.cpu().numpy(), num_data*epoch+iteration*training_config["batch_size"])

        if len(mean_module_loss) > 0:
        #if is_loss_learning(method):
            module_loss=sum(mean_module_loss)/len(mean_module_loss)
        else:
            module_loss=None
        # print(module_loss)
    return sum(mean_loss)/len(mean_loss), module_loss


def train(models, method, task, criterion, optimizers, lr_schedulers, train_dataset, validation_set, config, dataset_config, device, cycle_num, model_save_name="checkpoint.model"):
    experiment_path=config["experiment_path"]
    epoch_loss = config["epoch_loss"]
    if isinstance(models["task"],EnsembleModel) and config["ensemble_training"]:
        loss_list = []
        val_metric_list = {}
        config.experiment_config["experiment_path"] = experiment_path
        for i in range(len(models["task"])):
            ens_model_save_name = f'{model_save_name.split(".")[0]}_{i}_{model_save_name.split(".")[1]}'
            partial_loss_list, partial_val_metric_list = train_model({"task": models["task"][i]}, method, task, criterion, {"task":optimizers[f"task_{i}"]}, {"task":lr_schedulers[f"task_{i}"]}, train_dataset, validation_set, config, dataset_config, epoch_loss, device, experiment_path, cycle_num, ens_model_save_name)
            loss_list = partial_loss_list
            for key, value in partial_val_metric_list[0].items():
                val_metric_list[f"{key}_{i}"] = value
                val_metric_list[f"{key}"] = value
            checkpoint = torch.load(os.path.join(experiment_path, ens_model_save_name))
            models["task"][i].load_state_dict(checkpoint['model_state_dict']["task"])
        save_model(config["num_epochs"], models, optimizers, lr_schedulers, loss_list, experiment_path, model_save_name)
        val_metric_list=[val_metric_list]
            # metric[f"acc_{i}"] = ens_correct[i] / total
            # metric[f"f1_{i}"] = f1
            # metric[f"precision_{i}"] = precision
            # metric[f"recall_{i}"] = recall
            # if experiment_config.get("data_scenario", "") == DataUpdateScenario.osal_extending:
            #     extending_accuracy = (dataset_config["num_classes"] / dataset_config[
            #         "all_classes"]) * balanced_accuracy_score(ens_all_labels[:, i], ens_all_preds[:, i])
            #     metric[f"avg_acc_{i}"] = extending_accuracy
    else:
        loss_list, val_metric_list = train_model(models, method, task, criterion, optimizers, lr_schedulers, train_dataset, validation_set, config, dataset_config, epoch_loss, device, experiment_path, cycle_num, model_save_name)


    return loss_list, val_metric_list


def train_model(models, method, task, criterion, optimizers, lr_schedulers, train_dataset, validation_set, config, dataset_config, epoch_loss, device, experiment_path, cycle_num, model_save_name="checkpoint.model"):
    """
    #models, method, criterion, optimizers, schedulers, train_dataset,validation_set, training_config
    The functions the trains the model for num_epochs epochs and uses early stopping
    :param model:
    :param method:
    :param task:
    :param criterion:
    :param optimizer:
    :param data_loader:
    :param device:
    :param lr_scheduler:
    :param data_loader_test:
    :param path:
    :param num_epochs:
    :return:
    """
    stastistics = []
    stats_max = {}
    stats = {}
    metric = "val_acc"
    stats_max[metric] = None
    patience = 0
    track_metric = {
        TaskDef.classification: "acc",
        TaskDef.zeroclassification: "acc",
        TaskDef.semanticSegmentation: "mIoU",
        TaskDef.objectDetection: "mAP"
    }

    validation_metric = config.get("early_stopping", None)
    validation_metric_key = None
    if validation_metric is not None:
        patience_max = config["patience"]
        validation_metric_key = validation_metric.split("val_")[-1]
    loss_list = []
    val_metric_list = []


    gl_info("Start Training")
    with tqdm(range(config["num_epochs"]), leave=True, unit="Epochs", position=0) as epochs:
        for epoch in epochs:
            models["task"].train()
            if "module" in models:
                models['module'].train()
            loss, module_loss = train_epoch(models, method, task, criterion, optimizers, train_dataset, config,
                                            device, epoch, epoch_loss)

            if get_global_track_weights():
                track_all_weighs(models, epoch, cycle_num)

            if epoch % config["eval_frequencey"] == 0:
                train_metrics, _, _ = evaluate(models, method, task, criterion, train_dataset, config,
                                               dataset_config, device=device, cycle_num=cycle_num, mode="train")
                val_metrics, val_loss, val_mod_loss = evaluate(models, method, task, criterion, validation_set,
                                                               config, dataset_config, device=device,
                                                               cycle_num=cycle_num, mode="val")

                if validation_metric in EARLY_STOPPING_METRICS:
                    # change_train_flag(models, False)

                    # change_train_flag(models, True)
                    if config["early_stopping"] == "val_loss":
                        stats[metric] = val_loss
                        if stats_max[metric] is None:
                            stats_max[metric] = stats[metric]
                        if stats[metric] > stats_max[metric]:
                            if epoch > config.get("epoch_start", 0):
                                patience += 1
                        else:
                            save_model(epoch, models, optimizers, lr_schedulers, loss, experiment_path, model_save_name)
                            stats_max[metric] = stats[metric]
                            patience = 0
                    else:
                        stats[metric] = val_metrics[validation_metric_key]
                        if stats_max[metric] is None:
                            stats_max[metric] = stats[metric]
                        if stats[metric] < stats_max[metric]:
                            if epoch > config.get("epoch_start", 0):
                                patience += 1
                        else:
                            save_model(epoch, models, optimizers, lr_schedulers, loss, experiment_path, model_save_name)
                            stats_max[metric] = stats[metric]
                            patience = 0
                    if patience > patience_max:
                        break

                    stastistics.append(stats)
                    epochs.set_postfix_str(
                        f"val {track_metric[task]}: {val_metrics[track_metric[task]]:.6f} f1: {val_metrics.get('f1', -1.0):.6f} val loss: {val_loss}  loss:{str(loss)} P:{patience}/{patience_max}")
                else:
                    epochs.set_postfix_str(
                        f"val {track_metric[task]}: {val_metrics[track_metric[task]]:.6f} f1: {val_metrics.get('f1', -1.0):.6f} val loss: {val_loss}  loss:{str(loss)}")


                val_metric_list.append(val_metrics)
                loss_list.append(loss)
                write_stats_by_task(task, train_metrics, val_metrics, {"loss": loss, "modLoss": module_loss},
                                    {"loss": val_loss, "modLoss": val_mod_loss}, cycle_num, epoch)
            for s_name, scheduler in lr_schedulers.items():
                if scheduler:
                    scheduler.step()
    return loss_list, val_metric_list


def write_stats_by_task(task,train_metric,val_metric,loss,val_loss,cycle_num,epoch):
    global_write_scalar(f'Loss/C{cycle_num}-Epoch/train', loss["loss"], epoch)
    global_write_scalar(f'Loss/C{cycle_num}-Epoch/val', val_loss["loss"], epoch)
    gl_info(loss)
    gl_info(val_loss)
    if loss["modLoss"] is not None and val_loss["modLoss"] is not None:
        gl_info(val_loss)
        global_write_scalar(f'LossMod/C{cycle_num}-Epoch/train', loss["modLoss"], epoch)
        global_write_scalar(f'LossMod/C{cycle_num}-Epoch/val', val_loss["modLoss"], epoch)
    if task == TaskDef.classification or task == TaskDef.zeroclassification:
        write_cls_stats(train_metric,val_metric,cycle_num,epoch)
    elif task == TaskDef.semanticSegmentation:
        write_semseg_stats(train_metric, val_metric, cycle_num, epoch)


def write_cls_stats(train_metric,val_metric,cycle_num,epoch):
    global_write_scalar(f'Acc/C{cycle_num}-Epoch/val', val_metric["acc"], epoch)
    global_write_scalar(f'Acc/C{cycle_num}-Epoch/train', train_metric["acc"], epoch)
    global_write_scalar(f'F1/C{cycle_num}-Epoch/val', val_metric["f1"], epoch)
    global_write_scalar(f'F1/C{cycle_num}-Epoch/train', train_metric["f1"], epoch)


def write_semseg_stats(train_metric,val_metric,cycle_num,epoch):
    global_write_scalar(f'mIoU/C{cycle_num}-Epoch/val', val_metric["mIoU"], epoch)
    global_write_scalar(f'mIoU/C{cycle_num}-Epoch/train', train_metric["mIoU"], epoch)
    global_write_scalar(f'MIoU/C{cycle_num}-Epoch/val', val_metric["MIoU"], epoch)
    global_write_scalar(f'MIoU/C{cycle_num}-Epoch/train', train_metric["MIoU"], epoch)
    global_write_scalar(f'cmIoU/C{cycle_num}-Epoch/val', val_metric["cmIoU"], epoch)
    global_write_scalar(f'cmIoU/C{cycle_num}-Epoch/train', train_metric["cmIoU"], epoch)


def track_all_weighs(models,num_epoch,cycle):
    for modelname,model in models.items():
        d = model.state_dict()
        for k,v in d.items():
            if "weight" in k or "density_estimation" in k:
                global_write_histogram(f"C{cycle}-{modelname}-{k}",v.detach().cpu().numpy(),num_epoch)


def evaluate(models, method, task, criterion, dataloader, training_config, dataset_config, device, cycle_num=0, mode="val"):
    models["task"].eval()
    if "module" in models:
        models['module'].eval()
    evaluator = get_task_handler(task).evaluate #get_evaluator(task)
    metric, loss, module_loss = evaluator(device, dataloader, models, method, criterion, training_config, dataset_config, cycle_num, mode)
    return metric, loss, module_loss


def train_epoch_loss(models, method, task, criterion, optimizers, dataloader, training_config, device, epoch):
    print("Start Loss Training")
    # train_step = get_train_step(task)
    train_step = get_task_handler(task).train_step
    models['module'].train()
    # global iters
    mean_loss=[]
    mean_module_loss = []
    num_data=len(dataloader)
    iteration=0
    with tqdm(dataloader, leave=False, total=num_data, desc='Iterations', unit="Batch", position=1, disable=get_global_verbosity()) as iterator:
        for data in iterator:
            iteration+=1
            with torch.cuda.device(device):
                inputs = data[0].cuda()
                labels = data[1].cuda()

            # iters += 1 count global number of iterations
            with torch.autograd.set_detect_anomaly(True):
                optimizers['module'].zero_grad()
                loss, loss_module, _ = train_step(models, method, criterion, inputs, labels, epoch, training_config, epoch_loss=0)


                loss_module.backward()
                optimizers['module'].step()
            iterator.set_postfix_str(f"loss:{str(loss.data.cpu().numpy())}")
            mean_loss.append(loss.data.cpu().numpy())
            if loss_module is None:
                print("Check if 1 samples should be compared for loss")
            else:
                mean_module_loss.append(loss_module.data.cpu().numpy())
                global_write_scalar('LossMod/Iteration/trainL', loss_module,
                                    num_data * epoch + iteration * training_config["batch_size"])
            global_write_scalar('Loss/Iteration/trainL', loss, num_data * epoch + iteration * training_config["batch_size"])

        if is_loss_learning(method):
            module_loss = sum(mean_module_loss) / len(mean_module_loss)
        else:
            module_loss = None
    return sum(mean_loss)/len(mean_loss),module_loss


def train_loss(models, method, task, criterion, optimizers, lr_schedulers, train_dataset, validation_set, config, dataset_config, device, experiment_path, cycle_num, trained_epoch):
    """
    #models, method, criterion, optimizers, schedulers, train_dataset,validation_set, training_config
    The functions the trains the Faster RCNN for num_epochs epochs and uses early stopping
    :param model:
    :param optimizer:
    :param data_loader:
    :param device:
    :param lr_scheduler:
    :param data_loader_test:
    :param path:
    :param num_epochs:
    :return:
    """
    stastistics = []
    stats_max={}
    stats={}
    metric = "val_acc"
    patience = 0
    stats_max[metric]=0
    track_metric={
        TaskDef.classification: "acc",
        TaskDef.semanticSegmentation: "mIoU",
    }
    validation_metric=config.get("early_stopping", None)
    validation_metric_key=None
    if validation_metric is not None:
        patience_max = config["patience"]
        validation_metric_key=validation_metric.split("val_")[-1]
    loss_list = []
    val_metric_list = []

    with tqdm(range(trained_epoch, config["num_epochs"]+50), leave=True, unit="Epochs", position=0) as epochs:
        for epoch in epochs:
            models["task"].eval()
            models['module'].train()
            loss, module_loss = train_epoch_loss(models, method, task, criterion, optimizers, train_dataset, config, device, epoch)
            if get_global_track_weights():
                track_all_weighs(models, epoch, cycle_num)
            train_metric, _, _ = evaluate(models, method, task, criterion, train_dataset, config,dataset_config,  device=device,cycle_num=cycle_num,mode="train")
            val_metrics, val_loss, val_mod_loss = evaluate(models, method, task, criterion, train_dataset, config,dataset_config, device=device,cycle_num=cycle_num,mode="val")
            if validation_metric in ["val_acc", "val_loss", "val_mIoU", "val_mAP", "val_MIoU"]:
                # change_train_flag(models, False)

                # change_train_flag(models, True)
                # if training_config["early_stopping"] == "val_loss":
                stats[metric] = val_mod_loss
                if stats_max[metric] is None:
                    stats_max[metric] = stats[metric]
                if stats[metric] > stats_max[metric]:
                    if epoch > config.get("epoch_start", 0):
                        patience += 1
                else:
                    save_model(epoch, models, optimizers, lr_schedulers, loss, experiment_path)
                    stats_max[metric] = stats[metric]
                    patience = 0
                # else:
                #     stats[metric] = val_metrics[validation_metric_key]
                #     if stats_max[metric] is None:
                #         stats_max[metric] = stats[metric]
                #     if stats[metric] < stats_max[metric]:
                #         if epoch > training_config.get("epoch_start", 0):
                #             patience += 1
                #     else:
                #         save_model(epoch, models, optimizers, lr_schedulers, loss, experiment_path)
                #         stats_max[metric] = stats[metric]
                #         patience = 0
                if patience > patience_max:
                    break

                stastistics.append(stats)
                epochs.set_postfix_str(
                    f"val {track_metric[task]}: {val_metrics[track_metric[task]]:.6f} f1: {val_metrics.get('f1', -1.0):.6f} val loss: {val_loss}  loss:{str(loss)} P:{patience}/{patience_max}")
            else:
                epochs.set_postfix_str(
                    f"val {track_metric[task]}: {val_metrics[track_metric[task]]:.6f} f1: {val_metrics.get('f1', -1.0):.6f} val loss: {val_loss}  loss:{str(loss)}")
            if lr_schedulers["module"]:
                lr_schedulers["module"].step()
            val_metric_list.append(val_metrics)
            loss_list.append(loss)
            global_write_scalar(f'Loss/C{cycle_num}-Epoch/trainL', loss, epoch)
            global_write_scalar(f'Loss/C{cycle_num}-Epoch/valL', val_loss, epoch)
            global_write_scalar(f'Acc/C{cycle_num}-Epoch/valL', val_metrics["acc"], epoch)
            global_write_scalar(f'Acc/C{cycle_num}-Epoch/trainL', train_metric["acc"], epoch)
            global_write_scalar(f'F1/C{cycle_num}-Epoch/valL', val_metrics["f1"], epoch)
            global_write_scalar(f'F1/C{cycle_num}-Epoch/trainL', train_metric["f1"], epoch)
            global_write_scalar(f'LossMod/C{cycle_num}-Epoch/trainL', module_loss, epoch)
            global_write_scalar(f'LossMod/C{cycle_num}-Epoch/valL', val_mod_loss, epoch)
    # checkpoint = torch.load(path+".model")
    # models.load_state_dict(checkpoint['model_state_dict'])
    return loss_list, val_metric_list
