'''Train CIFAR10 with PyTorch.'''
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_class', default=10, type=int, help='num_class')
parser.add_argument('--job_id', default='local', type=str, help='job_id')
parser.add_argument('--load_model', '-r', action='store_true', help='load_model from checkpoint')
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--mode', default='train', type=str, help='Task to achieve')
parser.add_argument('--adv_epsilon', default=8, type=int)
parser.add_argument('--adv_step', default=20, type=int)
parser.add_argument('--adv_alpha', default=0.8, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--training_epoch', default=100, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_numpy_file', action='store_true', default=False)
parser.add_argument('--local', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--arch', default='resnet18', type=str, help='load_model_path')
parser.add_argument('--test_dir', default="/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_test", type=str)
parser.add_argument('--train_dir', default="/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train", type=str)
args = parser.parse_args()

import os
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from tqdm import tqdm
from ResNet import ResNet18
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics

import pandas as pd
import blobfile as bf
from PIL import Image
import numpy as np

from improved_diffusion.image_datasets import SimpleImageDataset as ImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# Training
def train(epoch, optimizer):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    train_bar = tqdm(trainloader)
    for pos_1, targets, ids in train_bar:
        inputs, targets = pos_1.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, args.training_epoch, train_loss/(batch_count), 100.*correct/total))

    return train_loss/(batch_count), 100.*correct/total

def adv():
    # print('\nEpoch: %d' % epoch)
    net.eval()
    epsilon = float(args.adv_epsilon) / 255.0
    alpha = float(args.adv_alpha) / 255.0
    
    test_bar = tqdm(testloader)
    for pos_1, targets, ids in test_bar:
        inputs, targets = pos_1.to(device), targets.to(device)

        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        for _step in range(args.adv_step):
            # print(_step)
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs = net(x_adv)
                loss = criterion(outputs, targets)
            # params_list = []
            # for params in net.parameters():
            #     if params.requires_grad:
            #         params_list.append(params)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, inputs - epsilon), inputs + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for img, idx in zip(x_adv, ids):
            save_image(img, testloader.dataset.local_images[idx].replace('cifar_test', 'cifar_test_{}_adv_test'.format(args.load_model_path)))

def adv_training(epoch, optimizer):
    # print('\nEpoch: %d' % epoch)
    epsilon = float(args.adv_epsilon) / 255.0
    alpha = float(args.adv_alpha) / 255.0
    
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    train_bar = tqdm(trainloader)

    for pos_1, targets, ids in train_bar:
        inputs, targets = pos_1.to(device), targets.to(device)

        net.eval()
        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        for _step in range(args.adv_step):
            # print(_step)
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs = net(x_adv)
                loss = criterion(outputs, targets)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, inputs - epsilon), inputs + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        net.train()
        optimizer.zero_grad()
        
        outputs = net(x_adv)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, args.training_epoch, train_loss/(batch_count), 100.*correct/total))

    return train_loss/(batch_count), 100.*correct/total

def test():
    data = np.load(args.test_dir)['arr_0'].astype(np.float32).transpose(0, 3, 1, 2)
    global best_acc
    net.eval()
    inputs = torch.tensor(data, device=device) / 255.0
    with torch.no_grad():
        
        outputs = net(inputs)
        # loss = criterion(outputs, targets)

        _, predicted = outputs.max(1)

        # print(predicted)

    bird_sample = []
    other_sample = []
    for i in range(len(data)):
        if predicted[i] == 0:
            bird_sample.append(data[i])
        else:
            other_sample.append(data[i])

    bird_sample = np.stack(bird_sample, axis=0).astype(np.uint8)
    grid_image = make_grid(torch.tensor(bird_sample), int(np.sqrt(len(bird_sample))) + 1, 2)#.float()

    im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
    im.convert('RGB').save(args.test_dir.replace(".npz", "bird.png"))

    other_sample = np.stack(other_sample, axis=0).astype(np.uint8)
    grid_image = make_grid(torch.tensor(other_sample), int(np.sqrt(len(other_sample))) + 1, 2)#.float()

    im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
    im.convert('RGB').save(args.test_dir.replace(".npz", "other.png"))
        
    return None

def evaluation(epoch, optimizer, save_name_pre):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    test_bar = tqdm(testloader)
    with torch.no_grad():
        for pos_1, targets, ids in test_bar:
            # print(targets)
            inputs, targets = pos_1.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            batch_count += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, args.training_epoch, test_loss/(batch_count), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not args.no_save:
            torch.save(state, './results/{}.pth'.format(save_name_pre))
        best_acc = acc
    return best_acc, test_loss/(batch_count), 100.*correct/total

print ("__name__", __name__)
if __name__ == '__main__':

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    args.num_class = 10

    save_name_pre = "local_supervised_class{}_{}".format(args.num_class, args.job_id)

    if not args.use_numpy_file:
        train_files = _list_image_files_recursively(args.train_dir)
        train_set = ImageDataset(train_files, transform_train)
    else:
        train_set = ImageDataset(args.train_dir, transform_train, use_numpy_file=args.use_numpy_file)
    # print(train_set[0][0].shape)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.mode != 'test':
        if not args.use_numpy_file:
            test_files = _list_image_files_recursively(args.test_dir)
            # print(test_files[0].replace('cifar_test', 'cifar_test_adv_linf'))
            test_set = ImageDataset(test_files, transform_test)
        else:
            test_set = ImageDataset(args.test_dir, transform_test, use_numpy_file=args.use_numpy_file)
        testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('==> Building model.. {}'.format(args.arch))

    net = ResNet18(args.num_class)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.load_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./results/supervised/{}.pth'.format(args.load_model_path))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training_epoch)

    results = {'train_loss': [], 'test_acc': [], 'train_acc': [], 'test_loss': [], 'best_test_acc': []}

    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch+args.training_epoch):
            train_loss, train_acc = train(epoch, optimizer)
            best_test_acc, test_loss, test_acc = evaluation(epoch, optimizer, save_name_pre)
            scheduler.step()
            # save statistics
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['best_test_acc'].append(best_test_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            # print(epoch, results)
            data_frame = pd.DataFrame(data=results, index=range(1, epoch+2))
            if not args.no_save:
                data_frame.to_csv('results/supervised/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
    elif args.mode == 'test':
        test()
    elif args.mode == 'adv_generate':
        adv()
    elif args.mode == 'adv_train':
        for epoch in range(start_epoch, start_epoch+args.training_epoch):
            train_loss, train_acc = adv_training(epoch, optimizer)
            best_test_acc, test_loss, test_acc = test(epoch, optimizer, save_name_pre)
            scheduler.step()
            # save statistics
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['best_test_acc'].append(best_test_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            # print(epoch, results)
            data_frame = pd.DataFrame(data=results, index=range(1, epoch+2))
            if not args.no_save:
                data_frame.to_csv('results/supervised/{}_statistics.csv'.format(save_name_pre), index_label='epoch')