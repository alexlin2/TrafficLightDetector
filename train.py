import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, device, dataloader, optimizer, epoch, losses_log):
    model.train()
    loss_sum_meter = AverageMeter()
    loss_classifier_meter = AverageMeter()
    loss_box_reg_meter = AverageMeter()
    loss_objectness_meter = AverageMeter()
    loss_rpn_box_reg_meter = AverageMeter()
    for i, (x, targets) in enumerate(dataloader):
        x = list(t.to(device) for t in x)
        labels = [{k: v.to(device) for k, v in t.items()}
                  for t in targets]
        loss_dict = model(x, labels)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        loss_sum = sum([v for _, v in loss_dict.items()])
        loss_sum_meter.update(loss_sum, len(x))
        loss_classifier_meter.update(loss_dict['loss_classifier'], len(x))
        loss_box_reg_meter.update(loss_dict['loss_box_reg'], len(x))
        loss_objectness_meter.update(loss_dict['loss_objectness'], len(x))
        loss_rpn_box_reg_meter.update(
            loss_dict['loss_rpn_box_reg'], len(x))
        if (i+1) % 10 == 0:
            log = ('[Train]Epoch: [{0}][{1}/{2}]\t'
                   'Loss_sum: {loss_sum.val: .6f}({loss_sum.avg: .6f})\t'
                   'Cls: {loss_classifier.val: .6f}({loss_classifier.avg: .6f})\t'
                   'Box: {loss_box_reg.val: .6f}({loss_box_reg.avg: .6f})\t'
                   'Obj: {loss_objectness.val: .6f}({loss_objectness.avg: .6f})\t'
                   'RPN {loss_rpn_box_reg.val: .6f}({loss_rpn_box_reg.avg: .6f})'.format(
                       epoch, i+1, len(dataloader), loss_sum=loss_sum_meter, loss_classifier=loss_classifier_meter,
                       loss_box_reg=loss_box_reg_meter, loss_objectness=loss_objectness_meter, loss_rpn_box_reg=loss_rpn_box_reg_meter))
            print(log)
            losses_log.append(loss_sum_meter.avg)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count