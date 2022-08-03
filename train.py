"""
Code adjusted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from newresnet import resnet101
from quantresnet import QuantResNet
import time
from datetime import datetime
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import warnings
from quantization import Quantizer
from companding import Compander
from utils import save_checkpoint, load_state_dict, kaiming_init, get_lr
from progressmeter import AverageMeter, ProgressMeter, Summary
from metrics import accuracy

# https://github.com/Tony-Y/pytorch_warmup/issues/5
warnings.filterwarnings("ignore", category=UserWarning)


## Transfroms

# CIFAR10 transfroms
cif10_transfroms = {
    'train': transforms.Compose([
        transforms.Resize(32),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                             std=[0.24703233, 0.24348505, 0.26158768])
    ]),
    'val': transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                             std=[0.24703233, 0.24348505, 0.26158768]),
    ]),
}

# CIFAR100 transfroms
cif100_transfroms = {
    'train': transforms.Compose([
            transforms.Resize(32),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                std=[0.2675, 0.2565, 0.2761]),
        ]),
    'val': transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]),
        ])
}

# ImageNet transfroms
imgnet_transfroms = {
    'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
}

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train for one epoch
def train(model, criterion, optimizer, scheduler, train_loader, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    # Iterate over data
    for i, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

            optimizer.step()
            if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
                scheduler.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg


# validate for one epoch
def validate(model, criterion, val_loader, epoch, args):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Val: [{}]".format(epoch))

    # Iterate over data.
    end = time.time()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    return top1.avg, top5.avg, losses.avg


def main_worker(args):

    best_acc1 = 0

    print(f'=> Running on device: {"cuda:0" if torch.cuda.is_available() else "cpu"}')

    # Load Datasets
    print(f"=> Loading dataset {args.dataset}...")
    cifar = True
    if args.dataset == 'cif10':
        num_classes = 10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=cif10_transfroms['train'])
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=False, transform=cif10_transfroms['val'])
    elif args.dataset == 'cif100':
        num_classes = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                     download=True, transform=cif100_transfroms['train'])
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                    download=False, transform=cif100_transfroms['val'])
    elif args.dataset == 'imagenette':
        num_classes = 10
        cifar = False
        train_dataset = torchvision.datasets.ImageFolder(root='./data/imagenette2/train',
                                                         transform=imgnet_transfroms['train'])
        val_dataset = torchvision.datasets.ImageFolder(root='./data/imagenette2/val',
                                                         transform=imgnet_transfroms['val'])
    else:
        num_classes = 1000
        cifar = False
        train_dataset = None
        val_dataset = torchvision.datasets.ImageFolder(root='./data/imagenet/val',
                                                     transform=imgnet_transfroms['val'])


    # Load Dataloaders
    train_loader = None
    val_loader = None

    print("=> Initializing dataloaders...")
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=0)

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=0)


    # Create model
    if args.split > 0:
        q = Quantizer(args.qbins)
        c = None
        if args.companding:
            c = Compander(args.qbins - 1)
        model = QuantResNet(args.split, c, q, pretrained=args.pretrained, num_classes=num_classes, cifar=cifar)
    else:
        model = resnet101(pretrained=args.pretrained, progress=False, num_classes=num_classes, cifar=cifar)
        # model = resnet101(pretrained=args.pretrained, progress=False, num_classes=num_classes)

    # Init weights if untrained
    if args.pretrained:
        print("=> Using pre-trained model")
    else:
        print("=> Creating untrained model...")
        model.apply(kaiming_init)


    # freeze all non fully-connected layers
    if args.freeze:
        print("=> Freezing layers...")
        for param in model.parameters():
            param.requires_grad = False

    # Replace fully-connected layer with appropriete one
    if not args.evaluate:
        if isinstance(model, QuantResNet):
            model.model_cloud.fc = nn.Linear(model.model_cloud.fc.in_features, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)


    # Move model to CUDA
    model.to(device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              args.lr,
    #                              betas=(0.9, 0.999),
    #                              weight_decay=args.decay)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=0.9,
                                weight_decay=args.decay,
                                nesterov=True)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=15,
    )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer=optimizer,
    #     milestones=[60, 120, 160],
    #     gamma=0.2
    # )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'...".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            load_state_dict(checkpoint, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'...".format(args.resume))


    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load)
            load_state_dict(checkpoint, model)
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(args.load, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'...".format(args.resume))


    # tensorboard
    train_writer = None
    val_writer = None
    now = datetime.now()
    str_now = now.strftime("%#d%b_%#H-%#M-%#S")
    if args.tensorboard:
        print("=> Tensorboard is enabled")
        train_writer = SummaryWriter(f"runs/{args.dataset}/{str_now}/train")
        val_writer = SummaryWriter(f"runs/{args.dataset}/{str_now}/val")


    if args.evaluate:
        validate(model, criterion, val_loader, "-", args)
        return

    if train_loader is None:
        print("Train dataloader is not specified. Cannot continue with training loop. Exiting...")
        return


    for epoch in range(args.start_epoch, args.epochs):
        n_iter = epoch * len(train_loader)

        # train for one epoch
        train_acc1, train_acc5, train_loss = train(model, criterion, optimizer, scheduler, train_loader, epoch, args)

        # evaluate on validation set
        val_acc1, val_acc5, val_loss = validate(model, criterion, val_loader, epoch, args)


        if not isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        # Save best model
        if args.save_model:
            dir_path = os.path.join(os.getcwd(), f'runs/{args.dataset}/{str_now}/checkpoints')
            save_checkpoint({
                'model': str(model.__class__),
                'epoch': epoch + 1,
                'dataset': args.dataset,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, dir_path)

        # Tensorboard display metrics at end of epoch
        if args.tensorboard:
            lr = get_lr(optimizer)
            train_writer.add_scalar('Learning rate', lr, n_iter)

            train_writer.add_scalar('Loss', train_loss, n_iter)
            train_writer.add_scalar('Acc@1', train_acc1, n_iter)
            train_writer.add_scalar('Acc@5', train_acc5, n_iter)

            val_writer.add_scalar('Loss', val_loss, n_iter)
            val_writer.add_scalar('Acc@1', val_acc1, n_iter)
            val_writer.add_scalar('Acc@5', val_acc5, n_iter)

        if args.dry_run:
            break

    # Log hyperparameters to Tensorboard
    if args.tensorboard:
        hparams_dict = {
            "model": str(model.__class__),
            "optim": str(optimizer.__class__),
            "sched": str(scheduler.__class__),
            "init lr": args.lr,
            "split": args.split,
            "qbins": args.qbins,
            "bsize": args.batch_size,
            "grad_clip": args.grad_clip,
            "weight_decay": args.decay,
            "epochs": args.epochs,
        }

        metric_dict = {
            "best_acc@1": best_acc1
        }

        train_writer.add_hparams(hparams_dict, metric_dict, run_name="./")

        train_writer.close()
        val_writer.close()


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--split', type=int, default=0, metavar='N',
                        help='block at which to split model in two (default: 0)')
    parser.add_argument('--qbins', type=int, default=6, metavar='N',
                        help='number of quantization bins (default: 6)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('-lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--decay', type=float, default=0.0, metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--grad-clip', type=float, default=None, metavar='N',
                        help='gradient clipping (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--companding', dest='companding', action='store_true',
                        help='use companding')
    parser.add_argument('-f', '--freeze', dest='freeze', action='store_true',
                        help='freeze all layers except the fully-connected one')
    parser.add_argument('--dataset', type=str, default="cif10", metavar='S',
                        help='Dataset to use for training and testing (default: cif10)')
    parser.add_argument('-e', '--evaluate', action='store_true', default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-ts', '--tensorboard', dest='tensorboard', action='store_true',
                        help='enable tensorboard summary writer')

    args = parser.parse_args()

    main_worker(args)


if __name__ == "__main__":
    main()
