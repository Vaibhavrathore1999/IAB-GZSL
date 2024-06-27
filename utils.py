import torch
import numpy as np
import shutil
import yaml
from tqdm import tqdm
import os
from datetime import datetime

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)

def save_args(args, log_path, argfile):
    try:
        shutil.copy(argfile, log_path)
    except:
        print('Config exists')

def create_log_folder():
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(logs_dir, current_time)
    os.makedirs(log_dir)

    print(f'MKDIR: {log_dir}')
    return log_dir

import os
import shutil

def copy_and_delete_folder(source_folder, target_folder):
    """
    Copies all files from the source folder to the target folder, then deletes the source folder.

    Args:
        source_folder (str): The path to the source folder.
        target_folder (str): The path to the target folder.
    """
    try:
        # Create the target folder if it doesn't exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Copy all files from the source folder to the target folder
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)
            shutil.copy2(source_file, target_file)

        # Delete the source folder
        shutil.rmtree(source_folder)
        print(f"Files copied from '{source_folder}' to '{target_folder}' and '{source_folder}' deleted.")
    except OSError as e:
        print(f"Error occurred: {e}")

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.acc_list = []
        self.epoch_list = []
    def update(self, it, acc):
        self.acc_list += [acc]
        self.epoch_list += [it]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.epoch_list += [it]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s

def cal_group_score(opt, logits):
    a, b = logits.size()
    print(a, b)
    output = logits.view(a, opt.nclasses, opt.Lp1).sum(2).view(a, opt.nclasses)
    return output

def compute_class_accuracy_total(opt,true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    acc = np.sum(true_label == predict_label) / len(true_label)
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    if opt.all:
        return acc
    else:
        return np.mean(acc_per_class)

def search_calibrated_stacking(opt,predict,lam):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    if not opt.NAA_test:
        output = predict.copy()
        seen_L = list(set(opt.test_seen_label.numpy()))
        output[:, seen_L] = output[:, seen_L] - lam
        return torch.from_numpy(output)
    else:
        output = predict.copy()
        for index in opt.test_seen_label.numpy():
            output[:, index * opt.Lp1:(index + 1) * opt.Lp1] \
                = output[:, index * opt.Lp1:(index + 1) * opt.Lp1] - lam
        return torch.tensor(output)
def test_gzsl(opt, model, testloader_seen,testloader_unseen, attribute,seen_classes,unseen_classes):
    with torch.no_grad():
        for i, (input, target, impath) in enumerate(testloader_seen):
            if opt.cuda:
                input = input.to(opt.device)
                target = target.to(opt.device)
            output = model(input, attribute)

            if i ==0:
                gt_s = target.cpu().numpy()
                logits_seen = output.cpu().numpy()
            else:
                gt_s = np.concatenate((gt_s,target.cpu().numpy()))
                logits_seen = np.vstack([logits_seen,output.cpu().numpy()])

        for i, (input, target, impath) in enumerate(testloader_unseen):
            if opt.cuda:
                input = input.to(opt.device)
                target = target.to(opt.device)
            output = model(input, attribute)

            if i ==0:
                gt_u = target.cpu().numpy()
                logits_unseen = output.cpu().numpy()
            else:
                gt_u = np.concatenate((gt_u,target.cpu().numpy()))
                logits_unseen = np.vstack([logits_unseen,output.cpu().numpy()])
    best_hm = 0
    best_seen = 0
    best_unseen = 0
    for cs in np.arange(0.000,0.5, 0.0001):
        logits_s = logits_seen.copy()
        logits_u = logits_unseen.copy()
        output_s = search_calibrated_stacking(opt,logits_s,cs)
        output_u = search_calibrated_stacking(opt,logits_u,cs)
        if opt.NAA_test:
            output_s = cal_group_score(opt, output_s)
            output_u = cal_group_score(opt, output_u)
        _, predicted_label_s = torch.max(output_s, 1)
        _, predicted_label_u = torch.max(output_u, 1)

        acc_all_s = compute_class_accuracy_total(opt, gt_s, np.array(predicted_label_s), seen_classes.numpy())
        acc_all_u = compute_class_accuracy_total(opt, gt_u, np.array(predicted_label_u), unseen_classes.numpy())
        H = (2 * acc_all_s * acc_all_u) / (acc_all_u + acc_all_s)
        #print(acc_all_s, acc_all_u, H)
        if H > best_hm:
            best_hm = H
            best_seen = acc_all_s
            best_unseen = acc_all_u
    return best_hm * 100,best_seen * 100,best_unseen * 100

def divide_into_groups(n=512, k=20):
    group = {}
    nums_per_group = n // k
    remainder = n % k

    start_num = 0
    for i in range(k):
        if i < remainder:
            end_num = start_num + nums_per_group + 1
        else:
            end_num = start_num + nums_per_group
        group[i] = list(range(start_num, end_num))
        start_num = end_num

    return group


