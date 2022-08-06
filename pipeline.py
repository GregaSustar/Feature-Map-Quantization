import os
import torch
import argparse
import torchvision

from companding import Compander
from quantization import Quantizer
from quantresnet import QuantResNet
from metrics import accuracy, bpp, CR, bitsize
from utils import load_state_dict, show
from arithmeticcoder import ArithmeticCoder
from progressmeter import Summary, ProgressMeter, AverageMeter
from train import cif10_transfroms, cif100_transfroms, imgnet_transfroms


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Eval settings')
    parser.add_argument('--split', type=int, default=0, metavar='N',
                        help='block at which to split model in two (default: 0)')
    parser.add_argument('--qbins', type=int, default=6, metavar='N',
                        help='number of quantization bins (default: 6)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--companding', dest='companding', action='store_true',
                        help='use companding')
    parser.add_argument('--adaptive', dest='adaptive', action='store_true',
                        help='use adaptive arithmetic coding')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='show before and after image of first pass through net')
    parser.add_argument('--dataset', type=str, default="cif10", metavar='S',
                        help='Dataset to use for training and testing (default: cif10)')
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()


    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


    print("=> Initializing model...")
    quant = Quantizer(args.qbins)
    coder = ArithmeticCoder("out.code", adaptive=args.adaptive)
    comp = None
    if args.companding:
        comp = Compander(args.qbins - 1)
    qnet = QuantResNet(args.split, comp, quant, pretrained=args.pretrained, num_classes=num_classes, cifar=cifar)


    if args.load:
        if os.path.isfile(args.load):
            checkpoint = torch.load(args.load)
            load_state_dict(checkpoint, qnet)
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(args.load, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'...".format(args.resume))


    # Start pipline
    qnet.eval()
    qnet.to(device)

    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    bitppx = AverageMeter('bpp', ':6.2f', Summary.AVERAGE)
    cratio = AverageMeter('CR', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [top1, top5, bitppx, cratio],
        prefix="Val: [-]")


    show_flag = args.show

    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        res = (inputs.shape[2], inputs.shape[3])

        with torch.no_grad():

            # Process input with first part of ResNet

            outputs = qnet.analysis(inputs)
            # Encode with arithmetic codin
            #coder.encode(outputs)

            # show before and after image
            if show_flag:
                if i == 98:
                    show(inputs[7])
                    show(qnet.edgepass(inputs)[7])
                    show(outputs[7], True)
                    show_flag = False

            # measure bpp and cr
            #_bpp = bpp(bitsize(coder.outputfile), res) / inputs.size(0)
            #cr = CR(_bpp)

            # Decode compressed file
            #outputs = coder.decode()
            # Process decoded tensor with second half of ResNet
            outputs = qnet.synthesis(outputs)

            # measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            #bitppx.update(_bpp, inputs.size(0))
            #cratio.update(cr, inputs.size(0))

            # measure elapsed time

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()
