import torch.nn as nn
import torch.nn.functional as F
import torch

def cos_similarity(fg,bg):
    fg = F.normalize(fg,dim=1)
    bg = F.normalize(bg,dim=1)
    sim = torch.mm(fg,bg.T)
    return torch.clamp(sim,min=0.0005,max=0.9995)
class SimMinLoss(nn.Module):  #Minimize Similarity
    def __init__(self,margin = 0.15,metric = 'cos',reduction = 'mean'):
        super(SimMinLoss,self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction
    def forward(self,bg_feat,fg_feat):
        sim = cos_similarity(bg_feat,fg_feat)
        loss = -torch.log(1-sim)
        return torch.mean(loss)
class SimMaxLoss(nn.Module):
    def __init__(self,metric = 'cos',alpha = 0.25,reduction = 'mean'):
        super(SimMaxLoss,self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
    def forward(self,feat):
        sim = cos_similarity(feat,feat)
        loss = -torch.log(sim)
        loss[loss<0] = 0
        _,indices = sim.sort(dim = 1)
        _,rank = indices.sort(dim = 1)
        rank = rank-1
        rank_weights = torch.exp(-rank.float()*self.alpha)
        loss = loss*rank_weights
        return torch.mean(loss)

def label_template(opt):
    neighbor_label = torch.zeros(opt.seen_classes * opt.Lp1, opt.seen_classes)
    for k in range(opt.seen_classes):
        neighbor_label[k * opt.Lp1:k * opt.Lp1 + opt.Lp1, k] = 1
    neighbor_label = neighbor_label.to(opt.device)
    return neighbor_label

def label_adjust(opt,scores,label_v):
    lt = label_template(opt)
    onehot_label = F.one_hot(label_v, opt.seen_classes)
    expanded_label = (onehot_label.unsqueeze(1) * lt.unsqueeze(0)).sum(-1)
    expanded_label = expanded_label.view(-1,opt.seen_classes,opt.Lp1)
    _, index = torch.topk(scores.view(-1, opt.seen_classes, opt.Lp1), dim=-1, k=opt.gamma)
    max_index = None
    for i in range(opt.gamma):
        if max_index is None:
            max_index = F.one_hot(index[:, :, i], opt.Lp1)
        else:
            max_index += F.one_hot(index[:, :, i], opt.Lp1)
    expanded_label = (max_index*opt.delta+1)*expanded_label
    other_labels = torch.ones_like(expanded_label)
    other_labels = other_labels*(1-(opt.gamma*opt.delta/(opt.Lp1-opt.gamma)))
    expanded_label = torch.where(expanded_label==1,other_labels,expanded_label)
    return expanded_label.view(-1,opt.Lp1*opt.seen_classes)

def cls_loss(label, logit): 
    correct = (label * logit).sum(-1)
    loss = -correct
    loss = torch.clamp(loss.neg(), min=1e-5)
    loss = loss.log().neg()
    return loss.mean()

def regr_loss(label, logit):
    correct = (label * logit).sum(-1)
    loss = -correct
    return loss.mean()

def contras(L_vars, fg, bg):
    loss1 = L_vars[0](bg)
    loss2 = L_vars[1](bg, fg)
    loss3 = L_vars[2](fg)
    return loss1 + loss2 + loss3

def Loss_fn(opt, label, logits, fg_feats, bg_feats, sim_map, model, realtrain, map = None):
    #set_gpu(opt)
    loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()

    expanded_label1 = label_adjust(opt, logits['cos'], label)
    expanded_label2 = label_adjust(opt, sim_map['cos'], label)

    L_vars = [SimMaxLoss(metric='cos', alpha=0.05).to(opt.device), SimMinLoss(metric='cos').to(opt.device),
                     SimMaxLoss(metric='cos', alpha=0.05).to(opt.device)]

    l1 = regr_loss(expanded_label1, logits['l2']) #l2 1.0
    l2 = cls_loss(expanded_label1, logits['cos']) #cos 0.1
    l3 = regr_loss(expanded_label1, sim_map['l2']) #
    l4 = contras(L_vars, fg_feats, bg_feats)  #vars 1.0
    l5 = regr_loss(expanded_label1, logits['cos_']) #

    l6 = regr_loss(expanded_label1, logits['cos'])
    l7 = criterion(sim_map['cos_'], label)   #
    loss = (loss + opt.alpha1 * l1 + opt.alpha2 * l2 + opt.alpha3 * l3 + opt.alpha4 * l4
            + opt.alpha5 * l5 + opt.alpha6 * l6 + opt.alpha7 * l7)

    if opt.additional_loss:
        weight_final = model.vars.activation_head.weight
        if realtrain:
            reg_loss = 5e-5 * weight_final.norm(2)
            loss += reg_loss

    return loss
