from regex import M
from utils.utils import setup_seed
from dataset.av_dataset import CremadDataset, VGGSound, AVMNIST
from dataset.text_dataset import M3AEClipDataset
from models.models import CLIPClassifier
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='KineticSound, CREMAD, TVMNIST, VGGSound,MVSA')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')

    return parser.parse_args()


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    loss_value_mm=[]
    loss_value_t=[]
    loss_value_v=[]

    cos_text=[]
    cos_visual=[]

    record_names_text = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('text' in name):
            record_names_text.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue

    for step, (images, text_input, label) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        images = images.to(device)
        text_input = text_input.to(device).long()
        label = label.to(device)
        
        out, out_v, out_t = model(images.float(), text_input)

        loss_mm = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_t = criterion(out_t, label)

        loss_value_mm.append(loss_mm.item())
        loss_value_t.append(loss_t.item())
        loss_value_v.append(loss_v.item())

        losses = [loss_mm, loss_v, loss_t]
        all_loss = ['both', 'visual', 'text']

        grads_text = {}
        grads_visual = {}

        for idx, loss_type in enumerate(all_loss):
            loss = losses[idx]
            loss.backward(retain_graph=True)

            if(loss_type == 'visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual if tensor_name in grads_visual[loss_type]])

            elif(loss_type == 'text'):
                for tensor_name, param in record_names_text:
                    if loss_type not in grads_text.keys():
                        grads_text[loss_type] = {}
                    if param.grad is not None:
                        grads_text[loss_type][tensor_name] = param.grad.data.clone()
                grads_text[loss_type]["concat"] = torch.cat([grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text if tensor_name in grads_text[loss_type]])

            else:
                for tensor_name, param in record_names_text:
                    if loss_type not in grads_text.keys():
                        grads_text[loss_type] = {}
                    if param.grad is not None:
                        grads_text[loss_type][tensor_name] = param.grad.data.clone() 
                grads_text[loss_type]["concat"] = torch.cat([grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text if tensor_name in grads_text[loss_type]])
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual if tensor_name in grads_visual[loss_type]])

            optimizer.zero_grad()
        
        this_cos_text = F.cosine_similarity(grads_text['both']["concat"], grads_text['text']["concat"], dim=0)
        this_cos_visual = F.cosine_similarity(grads_visual['both']["concat"], grads_visual['visual']["concat"], dim=0)

        text_task = ['both', 'text']
        visual_task = ['both', 'visual']

        text_k = [0, 0]
        visual_k = [0, 0]

        if(this_cos_text > 0):
            text_k[0] = 0.5
            text_k[1] = 0.5
        else:
            text_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_text[t].values()) for t in text_task])
        if(this_cos_visual > 0):
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])

        gamma = 1.5

        loss = loss_mm + loss_t + loss_v
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                layer = re.split('[_.]', str(name))
                if('head' in layer):
                    continue
                if('text' in layer):
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * text_k[0] * grads_text['both'][name] + 2 * text_k[1] * grads_text['text'][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if(diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if('visual' in layer):
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual['both'][name] + 2 * visual_k[1] * grads_visual['visual'][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if(diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

        optimizer.step()
        _loss += loss.item()

    return _loss / len(dataloader)


def valid(args, model, device, dataloader):

    n_classes = args.n_classes

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (images, text_input, label) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            text_input = text_input.to(device).long()
            label = label.to(device)
                
            prediction_all = model(images.float(), text_input)

            prediction = prediction_all[0]
            prediction_visual = prediction_all[1]
            prediction_text = prediction_all[2]

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_text = prediction_text[i].cpu().data.numpy()
                index_ma_text = np.argmax(ma_text)
                if index_ma_text == label[i]:
                    acc_t[label[i]] += 1.0

                ma_visual = prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num)


def main():

    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    model = CLIPClassifier(args)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    if args.dataset == 'CREMAD':
        train_dataset = CremadDataset(mode='train')
        test_dataset = CremadDataset(mode='test')
    elif args.dataset == 'VGGSound':
        train_dataset = VGGSound(mode='train')
        test_dataset = VGGSound(mode='test')
    elif args.dataset == 'AVMNIST':
        train_dataset = AVMNIST(mode='train')
        test_dataset = AVMNIST(mode='test')
    elif args.dataset == 'MVSA':
        train_dataset = M3AEClipDataset(args, mode='train')
        test_dataset = M3AEClipDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    # print("\n WARNING: Testing on a small dataset \n")
    # train_dataset = torch.utils.data.Subset(train_dataset, range(10))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(10))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    print(len(train_dataloader))

    if args.train:
        best_acc = -1

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(
                args, epoch, model, device, train_dataloader, optimizer, scheduler, None)

            acc, acc_v, acc_t = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_{}_epoch{}_batch{}_lr{}.pth'.format(
                    args.model, args.optimizer,  args.dataset, args.epochs, args.batch_size, args.learning_rate)

                saved_dict = {'saved_epoch': epoch,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)

                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_t: {:.4f}, Acc_v: {:.4f}".format(
                    batch_loss, acc, acc_t, acc_v))
            else:
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_t: {:.4f}, Acc_v: {:.4f}, Best Acc: {:.4f}".format(
                    batch_loss, acc, acc_t, acc_v, best_acc))


if __name__ == "__main__":
    main()
