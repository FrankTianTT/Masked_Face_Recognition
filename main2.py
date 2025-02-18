from dataset import customized_dataset
from model import FaceNet2
from util import get_Optimizer2, get_Scheduler, get_Sampler
from sampler import samples
from train import train2, load
from test import evalulate, test, predict
import numpy as np

import torch
import pandas as pd
from torch.utils.data import DataLoader
import multiprocessing
from arcface import ArcFaceLoss
from tqdm import tqdm

BATCH_SIZE = 128
NUM_WORKERS = multiprocessing.cpu_count()


def train():
    df_train = pd.read_csv('train.csv')
    df_eval1 = pd.read_csv('eval_same.csv')
    df_eval2 = pd.read_csv('eval_diff.csv')

    #######################################################################################
    #########################################config#######################################

    embedding_size = 512
    num_classes = df_train.target.nunique()
    weight_decay = 5e-4
    lr = 1e-1
    dropout = 0.4
    # resnet, effnet or None(IncepetionResNet)
    model_name = 'effnet'
    pretrain = False
    # 'arcface' or 'triplet'
    loss_fn = 'arcface'
    # global gem or None(avgerage pooling)
    pool = 'gem'
    # Cyclic or Step
    scheduler_name = 'multistep'
    # sgd or None(adam) or rmsprop
    optimizer_type = 'sgd'
    num_epochs = 25
    eval_every = 50
    # arcface loss seting
    arcface_s = 45
    arcface_m = 0.4
    class_weights_norm = 'batch'
    # focal or bce
    crit = "focal"
    # name of the model to be saved or loaded
    name = 'arcface_en.pth'
    #######################################################################################
    #######################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = customized_dataset(df_train, mode='train')
    eval_dataset1 = customized_dataset(df_eval1, mode='eval')
    eval_dataset2 = customized_dataset(df_eval2, mode='eval')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=False)
    eval_loader1 = DataLoader(eval_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=False)
    eval_loader2 = DataLoader(eval_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=False)

    # class_weights for arcface loss
    val_counts = df_train.target.value_counts().sort_index().values
    class_weights = 1 / np.log1p(val_counts)
    class_weights = (class_weights / class_weights.sum()) * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # arcface
    metric_crit = ArcFaceLoss(arcface_s, arcface_m, crit, weight=class_weights, class_weights_norm=class_weights_norm)

    # facenet = FaceNet2(num_classes=num_classes, model_name=model_name, pool=pool, embedding_size=embedding_size,
    #                    dropout=dropout, device=device, pretrain=pretrain)
    facenet = torch.load('./models/load.pth')
    facenet.model.freeze(3)

    optimizer = get_Optimizer2(facenet, metric_crit, optimizer_type, lr, weight_decay)  # optimizer
    scheduler = get_Scheduler(optimizer, lr, scheduler_name)  # scheduler
    # load previous trained model
    # facenet, optimizer, scheduler = load(name)
    # facenet.to(device)

    # train
    train2(facenet.to(device), train_loader, eval_loader1, eval_loader2, metric_crit, optimizer, scheduler, num_epochs,
           eval_every, num_classes, device, name)
    dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device, loss_fn)
    print('Distance threshold:', dist_threshold)


def detect_dataset():
    df_train = pd.read_csv('train.csv')
    df_eval1 = pd.read_csv('eval_same.csv')
    df_eval2 = pd.read_csv('eval_diff.csv')
    train_dataset = customized_dataset(df_train, mode='train')
    eval_dataset1 = customized_dataset(df_eval1, mode='eval')
    eval_dataset2 = customized_dataset(df_eval2, mode='eval')
    # train_dataset.detection()
    eval_dataset1.detection()
    eval_dataset2.detection()


def run():
    name = 'arcface_en.pth'
    loss_fn = 'arcface'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    df_test = pd.read_csv('assess.csv')
    df_eval1 = pd.read_csv('eval_same.csv')
    df_eval2 = pd.read_csv('eval_diff.csv')

    test_dataset = customized_dataset(df_test, mode='test')
    eval_dataset1 = customized_dataset(df_eval1, mode='eval')
    eval_dataset2 = customized_dataset(df_eval2, mode='eval')

    eval_loader1 = DataLoader(eval_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=False)
    eval_loader2 = DataLoader(eval_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                             drop_last=False)
    facenet, optimizer, scheduler = load(name)

    # dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device, loss_fn)
    # dist_threshold = 1.0299999713897705
    dist_threshold = 1.0699999332427979
    print('Distance threshold:', dist_threshold)
    # predict(facenet, test_dataset, test_loader, dist_threshold, device, loss_fn)
    #
    test(facenet, eval_loader1, dist_threshold, device, loss_fn)
    test(facenet, eval_loader2, dist_threshold, device, loss_fn)


if __name__ == "__main__":
    # detect_dataset()
    # train()
    run()
