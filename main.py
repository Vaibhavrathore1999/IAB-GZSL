from opt import get_opt
from dataset import DATA_LOADER,get_loader,map_label
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np
import sys
import json
import os
from model import IAB
from utils import Result, test_gzsl, divide_into_groups, load_args, save_args, create_log_folder, copy_and_delete_folder
import torch.optim as optim
from torch.autograd import Variable
from losses import Loss_fn
from tqdm import tqdm
from datetime import datetime
import logging
def main():
    #load configs
    opt = get_opt()
    if opt.dataset == 'CUB':
        opt.config = './configs/CUB.yaml'
    elif opt.dataset == 'AWA2':
        opt.config = './configs/AWA2.yaml'
    elif opt.dataset == 'SUN':
        opt.config = './configs/SUN.yaml'
    elif opt.dataset == 'FLO':
        opt.config = './configs/FLO.yaml'

    load_args(opt.config, opt)
    logpath = create_log_folder()
    save_args(opt, logpath, opt.config)
    print('opt:', opt)

    logging.basicConfig(
        filename= os.path.join(logpath,'train_{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    #cuda
    cudnn.benchmark = True
    device = opt.device
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    #dataloader
    data = DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label
    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")
    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # load attribute groups
    if opt.random_grouping:
        group_dic = divide_into_groups(opt.att_size, opt.Lp1 - 1)
    else:
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, opt.group_path)))
        opt.Lp1 = len(group_dic)+1

    # prepare the attribute labels and model
    class_attribute = data.attribute
    print('Groups nubmer:', len(group_dic))
    print('Create Model...')
    model = IAB(opt, group_dic, data)

    if torch.cuda.is_available():
        model.to(device)
        class_attribute = class_attribute.to(device)

    result_zsl = Result()
    result_gzsl = Result()

    print('-----------------training-------------------')
    for epoch in range(opt.nepoch):
        current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
        realtrain = epoch > opt.pretrain_epoch
        if epoch <= opt.pretrain_epoch:
            model.fine_tune(False, False)
            model.NAA.beta_finetune(opt.train_beta)
            model.vars_finetune(opt.unfix_vars)

            model_params = [param for name, param in model.named_parameters() if param.requires_grad]
            optim_params = [{'params': model_params}]
            optimizer = optim.Adam(optim_params, lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
        else:
            model.fine_tune(opt.unfix_low, opt.unfix_high)
            model.NAA.beta_finetune(opt.train_beta)
            model.vars_finetune(True)

            model_params = [param for name, param in model.named_parameters() if param.requires_grad]
            optim_params = [{'params': model_params}]
            optimizer = optim.Adam(optim_params, lr=current_lr, betas=(opt.beta1, 0.999))

        batch = len(trainloader)
        model.train()
        to_average = []
        i = batch - 1
        for i, (batch_input, batch_target, impath) in enumerate(trainloader):
            model.zero_grad()
            batch_target = map_label(batch_target, data.seenclasses)
            image = Variable(batch_input)
            label = Variable(batch_target)
            if opt.cuda:
                image = image.to(device)
                label = label.to(device)
            top_k, sim_map, fg_feature, bg_feature, map = model(image, class_attribute)
            loss = Loss_fn(opt, label, top_k, fg_feature, bg_feature, sim_map, model, realtrain, map)
            loss.backward()
            optimizer.step()
            to_average.append(loss.item() / opt.gamma)
        print('\n[Epoch %d]'% (epoch + 1),'Loss=',sum(to_average) / len(to_average))
        if (i + 1) == batch or (i + 1) % 200 == 0:
            print('-----------------testing-------------------')
            model.eval()
            if opt.NAA_test:
                attributes = class_attribute.clone()
            else:
                attributes = class_attribute.clone().T
            acc_GZSL_H, acc_GZSL_seen, acc_GZSL_unseen = test_gzsl(opt, model, testloader_seen, testloader_unseen, attributes, data.seenclasses, data.unseenclasses)

            if acc_GZSL_H > result_gzsl.best_acc:
                model_save_path = os.path.join(logpath,'{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                torch.save(model.state_dict(), model_save_path)
                print('model saved to:', model_save_path)

            result_gzsl.update_gzsl(epoch + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H)
            print('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                  '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                  format(epoch + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U,
                         result_gzsl.best_acc_S,
                         result_gzsl.best_acc, result_gzsl.best_iter))
            logging.info('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                  '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                  format(epoch + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U,
                         result_gzsl.best_acc_S,
                         result_gzsl.best_acc, result_gzsl.best_iter))

    FinishedPath = 'HM:{}_S:{}_U:{}'.format(result_gzsl.best_acc,result_gzsl.best_acc_U,result_gzsl.best_acc_S)
    copy_and_delete_folder(logpath, os.path.join('logs',FinishedPath))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # print('Best AUC achieved is ', best_auc)
        # print('Best HM achieved is ', best_hm)
        print('quit')

