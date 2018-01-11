import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        batch_size = inputs.size(0)
        if not opt.no_cuda:
            targets = targets.cuda(async=True)

        loss_sum, acc_sum = 0, 0
        optimizer.zero_grad()

        # Split batch into multiple splits and calculate gradients separately and accumulate
        # torch.chunk()
        # input_splits = [x for x in np.array_split(inputs, opt.batch_split) if x.size > 0]
        # target_splits = [x for x in np.array_split(targets, opt.batch_split) if x.size > 0]
        input_splits = torch.chunk(inputs, opt.batch_split)
        target_splits = torch.chunk(targets, opt.batch_split)
        for input_split, target_split in zip(input_splits, target_splits):
            split_size = input_split.size(0)

            input_split = Variable(input_split)
            target_split = Variable(target_split)
            outputs = model(input_split)
            # Rescale the loss so that the accumulated loss equals to loss of the original batch calculated as a whole
            loss = criterion(outputs, target_split) * split_size / batch_size
            acc = calculate_accuracy(outputs, target_split) * split_size / batch_size

            loss_sum += loss.data[0]
            acc_sum += acc

            loss.backward()
        # Update weights with the accumulated gradients from the whole batch
        optimizer.step()

        losses.update(loss_sum, batch_size)
        accuracies.update(acc_sum, batch_size)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch, i + 1, len(data_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
