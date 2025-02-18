import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('test_result.png')


def result(model, dataloader, device, loss_fn='triplet'):
    dist = []
    with torch.no_grad():
        model.eval()
        for _, eval_data in enumerate(tqdm(dataloader)):
            eval_image = eval_data['image'].to(device)
            eval_out = model(eval_image)
            eval_pair = eval_data['pair_image'].to(device)
            eval_pait_out = model(eval_pair)
            if loss_fn == 'arcface':
                distance = torch.norm(eval_out['embeddings'] - eval_pait_out['embeddings'], dim=1)
            else:
                distance = torch.norm(eval_out - eval_pait_out, dim=1)
            dist.append(list(distance.cpu().numpy()))

    new_dist = []
    for i in range(len(dist)):
        for j in range(len(dist[i])):
            new_dist.append(dist[i][j])
    dist = np.asarray(new_dist)

    return dist


# get the distance threshold
def evalulate(model, eval_loader1, eval_loader2, device, loss_fn='triplet'):
    # same target pairs
    dist1 = result(model, eval_loader1, device, loss_fn)
    # diff target pairs
    dist2 = result(model, eval_loader2, device, loss_fn)
    same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),
                                            np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
    diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),
                                            np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
    difference = same_hist[0] - diff_hist[0]
    difference[:same_hist[0].argmax()] = np.Inf
    difference[diff_hist[0].argmax():] = np.Inf
    return (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1]) / 2


def test(model, test_loader, dist_threshold, device, loss_fn='triplet'):
    label = []
    pred = []
    dist = []
    with torch.no_grad():
        model.eval()
        for _, test_data in enumerate(tqdm(test_loader)):
            test_image = test_data['image'].to(device)
            test_target = test_data['target']
            test_out = model(test_image)
            test_pair = test_data['pair_image'].to(device)
            test_pair_target = test_data['pair_target']
            test_pait_out = model(test_pair)
            if loss_fn == 'arcface':
                distance = torch.norm(test_out['embeddings'] - test_pait_out['embeddings'], dim=1)
            else:
                distance = torch.norm(test_out - test_pait_out, dim=1)
            # dist.append(list(distance.cpu().numpy()))
            label.append(list((test_target == test_pair_target).cpu().numpy()))
            pred.append(list((distance <= dist_threshold).cpu().numpy()))

    new_label = []
    new_pred = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            new_label.append(label[i][j])
            new_pred.append(pred[i][j])

    # if the image pairs are different class, equal to 0 else 1
    new_pred = [0 if i == False else 1 for i in new_pred]
    # if the pred image less than threshold (same class) is 1 else 0
    new_label = [0 if i == False else 1 for i in new_label]
    new_pred = np.array(new_pred)
    new_label = np.array(new_label)
    num_true = np.sum(new_pred == new_label)
    acc = num_true / len(new_label)
    print('Accuracy:', acc)
    # 0 is negative, 1 is positive
    print(classification_report(new_label, new_pred))
    cm = confusion_matrix(new_label, new_pred)
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(cm, [0, 1])


def predict(model, test_dataset, test_loader, dist_threshold, device, loss_fn='triplet'):
    label = []
    pred = []
    dist = []
    with torch.no_grad():
        model.eval()
        for _, test_data in enumerate(tqdm(test_loader)):
            test_image = test_data['image'].to(device)
            test_out = model(test_image)
            test_pair = test_data['pair_image'].to(device)
            test_pait_out = model(test_pair)
            if loss_fn == 'arcface':
                distance = torch.norm(test_out['embeddings'] - test_pait_out['embeddings'], dim=1)
            else:
                distance = torch.norm(test_out - test_pait_out, dim=1)
            # dist.append(list(distance.cpu().numpy()))
            pred.append(list((distance <= dist_threshold).cpu().numpy()))

    new_pred = []
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            new_pred.append(pred[i][j])

    # os.makedirs("pred")
    # for i in range(200):
    #     path = os.path.join("pred", "{:03d}.png".format(i))
    #     test_dataset.show(i, new_pred[i], path)
    result = dict([("{}".format(i + 1), int(new_pred[i])) for i in range(len(test_dataset))])
    print(result)
    import json
    result = json.dumps(result)
    with open("result_en.json", "w") as f:
        f.write(result)

    # True, False, True, True, False, True, False, True, True, False, False, False, True, False, True, False, False, False, False, False
