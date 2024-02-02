from collections import defaultdict
from curses import init_pair

from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
from util.data_process import DataTool
import os
from core.HAKE import HAKE
from core.RotPro import RotPro
from core.TransE import TransE
from core.DTransE import DTransE
from core.TuckER import TuckER
from core.DistMult import DistMult
from core.RotatE import RotatE
from core.SpereLevel import SphereLevel
from core.HAKEpro import HAKEpro
from core.BoxLevel import BoxLevel
from core.PairRE import PairRE
from core.SemRotatE import SemRotatE
import numpy as np
import json
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator,OneShotIterator,NagativeSampleNewDataset
from loss import NSSAL,MRL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random
import yaml

def caculate_sphere(embedding):
    center = torch.mean(embedding, dim = 0)
    max_edge,_ = torch.max(embedding,dim=0)
    r = torch.norm(center-max_edge)
    return center,r

def loss_of_(c1,r1,c2,r2):
    d = torch.norm(c1-c2,dim=-1)
    return torch.relu(r1+r2-d)


def caculate_loss(type2ids, embedding):
    types = ['<wordnet_person_100007846>','<wordnet_abstraction_100002137>','<yagoGeoEntity>','<wordnet_organization_108008335>',
        '<wordnet_artifact_100021939>', '<wordnet_building_102913152>']
    
    per_embedding = embedding(type2ids[types[0]])
    abs_embedding = embedding(type2ids[types[1]])
    geo_embedding = embedding(type2ids[types[2]])
    org_embedding = embedding(type2ids[types[3]])
    art_embedding = embedding(type2ids[types[4]])
    bui_embedding = embedding(type2ids[types[5]])

    per_c,per_r = caculate_sphere(per_embedding)
    abs_c,abc_r = caculate_sphere(abs_embedding)
    geo_c,geo_r = caculate_sphere(geo_embedding)
    org_c,org_r = caculate_sphere(org_embedding)
    art_c,art_r = caculate_sphere(art_embedding)
    bui_c,bui_r = caculate_sphere(bui_embedding)

    loss = 0
    loss += loss_of_(per_c,per_r,abs_c,abc_r)
    loss += loss_of_(per_c,per_r,geo_c,geo_r)
    loss += loss_of_(per_c,per_r,org_c,org_r)
    loss += loss_of_(per_c,per_r,art_c,art_r)
    loss += loss_of_(per_c,per_r,bui_c,bui_r)

    loss += loss_of_(abs_c,abc_r,geo_c,geo_r)
    loss += loss_of_(abs_c,abc_r,org_c,org_r)
    loss += loss_of_(abs_c,abc_r,art_c,art_r)
    loss += loss_of_(abs_c,abc_r,bui_c,bui_r)

    loss += loss_of_(geo_c,geo_r,org_c,org_r)
    loss += loss_of_(geo_c,geo_r,art_c,art_r)
    loss += loss_of_(geo_c,geo_r,bui_c,bui_r)

    loss += loss_of_(org_c,org_r,art_c,art_r)
    loss += loss_of_(org_c,org_r,bui_c,bui_r)

    loss += loss_of_(art_c,art_r,bui_c,bui_r)


    return loss


def read_disjonit(file):
    type2ids = {}
    with open(file,encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            etype,eid = line.strip().split(':')
            etype = etype.strip()
            eids = eid.strip().split('\t')
            type2ids[etype] = list(map(int,eids)) 
    return type2ids

def read_tree_nag(dic_file, entity2id):
    
    typeDiff = []
    for line in open(os.path.join(dic_file,'typeDiff.txt'),'r',encoding='utf-8'):
        data  = line.strip().split('\t')
        data = list(map(float, data))
        typeDiff.append(data)
    
    entity2Typeid = {}
    with open(os.path.join(dic_file,'entity2typeid.txt'),'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            entity, typeId = line.strip().split('\t')
            entity2Typeid[entity2id[entity]] =int(typeId)
    

    entityIds = entity2Typeid.keys()
    entityIds = sorted(entityIds)
    typeIds = []
    for entityId in entityIds:
        typeIds.append(entity2Typeid[entityId])
    
    return typeDiff, typeIds



def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)


def train_with_disjoint_negative(train_iterator,model,loss_function,cuda, args,disjoint):
    positive_sample,negative_sample, subsampling_weight, mode, weight = next(train_iterator)
    # positive_sample,negative_sample, subsampling_weight, mode= next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
        weight = weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode)
    else:
        negative_score = model(negative_sample,r, t,mode)
    positive_score = model(h,r,t)
   
    loss = loss_function(positive_score, negative_score,subsampling_weight,weight)
    # loss = loss_function(positive_score, negative_score,subsampling_weight)

    log = {
        '_loss': loss.item(),
    }
    return log, loss    


def train_step(train_iterator,model,loss_function,cuda, args,disjoint):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode)
    else:
        negative_score = model(negative_sample,r, t,mode)
    positive_score = model(h,r,t)
   
    loss = loss_function(positive_score, negative_score,subsampling_weight)
    reg = args.loss_weight * caculate_loss(disjoint,model.entity_embedding)
    loss_1 = loss  + reg

    log = {
        '_loss': loss.item(),
         'regul': reg.item()
    }
    return log, loss_1


def train_step_old(train_iterator,model,loss_function,cuda, args,disjoint=None):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode)
    else:
        negative_score = model(negative_sample,r, t,mode)
    positive_score = model(h,r,t)
   
    loss = loss_function(positive_score, negative_score,subsampling_weight)
    # loss = loss_function(positive_score, negative_score)
#    loss +=model.caculate_constarin(args.gamma_m, args.beta, args.alpha)
  #  reg = args.loss_weight *model.caculate_constarin()
    loss_1 = loss #  + reg

    log = {
        '_loss': loss.item(),
        #'regul': reg.item()
    }
    return log, loss_1




def test_step_f(model, test_triples, all_true_triples,nentity,nrelation,cuda=True,inverse=False, onType=None,train_type=None):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'hr_t'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'h_rt'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    if not onType is None:
        if onType == 'head':
            test_dataset_list = [test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    else:
        if not inverse:
            test_dataset_list = [test_dataloader_tail,test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0
    # print(total_steps)
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                batch_size = positive_sample.size(0)
                if cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()
                    
                h = positive_sample[:,0]
                r = positive_sample[:,1]
                t = positive_sample[:,2] 
                if mode == 'hr_t':
                    negative_score = model(h,r, negative_sample,mode='hr_all',train_type=train_type)
                    positive_arg = t
                else:
                    negative_score = model(negative_sample,r,t,mode='all_rt',train_type=train_type)
                    positive_arg = h
                # 
                score = negative_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)
                
                for i in range(batch_size):
                    count = count + 1
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                step += 1
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics["Test Count"] = count
    return metrics

def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--valid', action='store_true', help='valid model')
    
    parser.add_argument('--max_step', type=int,default=200001, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)
    parser.add_argument("--neg_size",type=int, default=256)
    parser.add_argument("--gamma", type=float, default=20)
    parser.add_argument("--adversial_temp", type=float, default=0.5)

    parser.add_argument("--dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=50000)

    parser.add_argument("--loss_function", type=str)

    # HAKE 模型的混合权重
    parser.add_argument("--mode_weight",type=float,default=0.5)
    parser.add_argument("--phase_weight",type=float,default=0.5)

    parser.add_argument("--g_type",type=int,default=5)
    parser.add_argument("--g_level",type=int,default=5)

    parser.add_argument("--model",type=str)
    parser.add_argument("--init",type=str)

    parser.add_argument("--g_mode",type=float,default=0.5)
    parser.add_argument("--g_phase",type=float,default=0.5)

    # RotPro 约束参数配置
    parser.add_argument("--gamma_m",type=float,default=0.000001)
    parser.add_argument("--alpha",type=float,default=0.0005)
    parser.add_argument("--beta",type=float,default=1.5)
    parser.add_argument("--train_pr_prop",type=float,default=1)
    parser.add_argument("--loss_weight",type=float,default=1)
    parser.add_argument("--config_name",type=str,default="None")

    # 选择数据集
    parser.add_argument("--level",type=str,default='ins')
    parser.add_argument("--data_inverse",action='store_true')

    return parser.parse_args(args)

def save_embedding(emb_map,file):
    for key in emb_map.keys():
        np.save(
            os.path.join(file,key),emb_map[key].detach().cpu().numpy()
        )

# 修改了负采样技术，增加了增加了disjoint 的损失
def build_dataset_new_nagtive(data_path,args):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    
    disjoint = read_disjonit("/home/skl/yl/data/YAGO3-1668k/yago_insnet/type2id.txt")
    typeDiff, entity2Typeid =  read_tree_nag("/home/skl/yl/data/YAGO3-1668k/yago_insnet/",on.entity2id)
   
    on_train_t = DataLoader(NagativeSampleNewDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t',disjoint,typeDiff, entity2Typeid),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleNewDataset.collate_fn
    )
    if(not args.data_inverse):
        on_train_h = DataLoader(NagativeSampleNewDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt',disjoint,typeDiff, entity2Typeid),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleNewDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
        
    on_ite = {
        "train": on_train_iterator,
        "valid": None,
        "test": None
    }
    return on,on_ite

def build_dataset(data_path,args):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
   
    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
        
    on_valid = DataLoader(NagativeSampleDataset(on.valid, on.nentity, on.nrelation, 2, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_test = DataLoader(NagativeSampleDataset(on.test, on.nentity, on.nrelation, 2, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    
    on_valid_ite = OneShotIterator(on_valid)
    on_test_ite = OneShotIterator(on_test)
    on_ite = {
        "train": on_train_iterator,
        "valid": on_valid_ite,
        "test": on_test_ite
    }
    return on,on_ite

def getModel(model_name):
    model_dic={
        'TransE':TransE,
        'HAKE':HAKE,
        "RotatE":RotatE,
        'RotPro':RotPro,
        'HAKEpro':HAKEpro,
        'BoxLevel':BoxLevel,
    }
    return model_dic[model_name]


def test_mapping(step,dataset,cuda):
    data_path ="/home/skl/yl/FB15k/mapping"
    test_file = ['121.txt','12n.txt','n21.txt','n2n.txt']
    for file_name in test_file:
        file = os.path.join(data_path,file_name)
        triples = DP.read_triples(file,entity2id=dataset.entity2id,relation2id=dataset.relation2id)
        logging.info('Test at patter: %s' % file_name[:-4])
        logging.info('predicate Head : %s' % file_name[:-4])
        metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='head')
        logset.log_metrics('Valid ',step, metrics)
        logging.info('predicate Tail : %s' % file_name[:-4])
        metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='tail')
        logset.log_metrics('Valid ',step, metrics)


def read_disjoint_with_typedic(file,nentity):
    type2ids = {}
    id2type = [0 for i in range(nentity)]
    types = ['<wordnet_person_100007846>','<wordnet_abstraction_100002137>','<yagoGeoEntity>','<wordnet_organization_108008335>',
            '<wordnet_artifact_100021939>', '<wordnet_building_102913152>']
    with open(file,encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            etype,eid = line.strip().split(':')
            etype = etype.strip()
            eids = eid.strip().split('\t')
            type2ids[etype] = list(map(int,eids)) 

    etypes = sorted(type2ids.keys())
    count = 0
    for i in range(len(etypes)):
        etype = etypes[i]
        count += len(type2ids[etype])
        for eid in type2ids[etype]:
            id2type[eid] = i 
    type2IdsList = []
    print("Have TypeId total: %d" % count)
    for type in types:
        type2IdsList.append(type2ids[type])

    return type2IdsList,id2type


def split_triples(triples,relation2id):
    # 将关系分类处理，确定不同类型的关系，然后将训练数据分为两类，构造四个数据集
    inter_relation = ['isAffiliatedTo','isConnectedTo','influences','dealsWith','hasChild','isMarriedTo','hasAcademicAdvisor']
    inter_relationid = []
    for rel in inter_relation:
        inter_relationid.append(relation2id[rel])

    inter_triples = []
    outer_triples = []
    for h,r,t in triples:
        if r in inter_relationid:
            inter_triples.append((h,r,t))
        else:
            outer_triples.append((h,r,t))
    return inter_triples, outer_triples

def handle_things_for_rotate(data_path,args):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    
    inter_triples, outer_triples = split_triples(on.train,on.relation2id)
    print("triple size of inter: %d " % len(inter_triples))
    print("triple size of outer: %d " % len(outer_triples))
    type2eids, eid2typeid = read_disjoint_with_typedic("/home/skl/yl/data/YAGO3-1668k/yago_insnet/type2id.txt",on.nentity)
    print(len(eid2typeid))
    inter_train_t = DataLoader(NagativeSampleDataset(inter_triples, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        inter_train_h = DataLoader(NagativeSampleDataset(inter_triples, on.nentity, on.nrelation, n_size, 'h_rt'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        inter_train_iterator = BidirectionalOneShotIterator(inter_train_t, inter_train_h)
    else:
        inter_train_iterator = OneShotIterator(inter_train_t)

    outer_train_t = DataLoader(NagativeSampleDataset(outer_triples, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        outer_train_h = DataLoader(NagativeSampleDataset(outer_triples, on.nentity, on.nrelation, n_size, 'h_rt'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        outer_train_iterator = BidirectionalOneShotIterator(outer_train_t, outer_train_h)
    else:
        outer_train_iterator = OneShotIterator(outer_train_t)
        
    on_ite = {
        "inter": inter_train_iterator,
        "outer": outer_train_iterator,
        "test": None
    }
    return on,on_ite,type2eids, eid2typeid

def train_rel_split_step(train_iterator,model,loss_function,cuda, args,train_type='inter'):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode,train_type=train_type)
    else:
        negative_score = model(negative_sample,r, t,mode,train_type=train_type)
    positive_score = model(h,r,t,train_type=train_type)
   
    loss = loss_function(positive_score, negative_score,subsampling_weight)
    reg = args.loss_weight * caculate_loss(disjoint,model.entity_embedding)
    loss_1 = loss  + reg

    log = {
        '_loss': loss.item(),
         'regul': reg.item()
    }
    return log, loss_1
# 记录了没有hit10的测试用例
def test_print_bad_case(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'hr_t'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'h_rt'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    if not onType is None:
        if onType == 'head':
            test_dataset_list = [test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    else:
        if not inverse:
            test_dataset_list = [test_dataloader_tail,test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0

    baseCaseList = []
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                batch_size = positive_sample.size(0)
                if cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()
                    
                h = positive_sample[:,0]
                r = positive_sample[:,1]
                t = positive_sample[:,2] 
                if mode == 'hr_t':
                    negative_score = model(h,r, negative_sample,mode=mode)
                    positive_arg = t
                else:
                    negative_score = model(negative_sample,r,t,mode=mode)
                    positive_arg = h
                
                # 将正样本的score - 1

                score = negative_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                # 对score 进行排序：找到对应的score 
                
                for i in range(batch_size):
                    count = count + 1
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1
                    ranking = 1 + ranking.item()
                    if ranking > 10: # this is a bad Case, record it
                        badCase = {}
                        badCase['mode'] =mode
                        badCase['true'] = (h[i].cpu().numpy().tolist(),r[i].cpu().numpy().tolist(),t[i].cpu().numpy().tolist())
                        badCase['top10'] =negative_sample[i][argsort[i,:10]].cpu().numpy().tolist()
                        baseCaseList.append(badCase)

                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                step += 1
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics["Test Count"] = count
    return metrics,baseCaseList


def overight_args(args, baseConfig, modelConfig):
    args.cuda = baseConfig['cuda']
    args.train = baseConfig['do_train']
    args.test = baseConfig['test']
    args.max_step = modelConfig['max_step']
    args.batch_size = baseConfig['batch_size']
    args.test_step = baseConfig['test_step']
    args.dim = modelConfig['dim']
    args.lr = modelConfig['lr']
    args.decay = modelConfig['decay']
    args.gamma = modelConfig['gamma']
    args.loss_weight = modelConfig['loss_weight']

    args.level = 'ins'
    args.train_type = 'rel_spl'


if __name__=="__main__":
    # 读取4个数据集
    args = set_config()

    with open('./config/repu.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.config_name]

    overight_args(args, baseConfig,modelConfig)

    cuda = args.cuda
    init_step  = 0
    save_steps = 10000
    init_path = args.init
    max_step   = args.max_step
    batch_size = args.batch_size
    test_step = args.test_step
    dim = args.dim
    lr = args.lr
    decay = args.decay
    warm_up_steps = args.warm_up_step

    g_ons = args.gamma
    n_size = modelConfig['n_size']
    log_steps = 100


    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    args.save_path = root_path
    if args.level == 'on':
        # test_step = 4000
        data_path ="/home/skl/yl/data/YAGO3-1668k/yago_ontonet"
    elif args.level == 'ins_on':
        test_step = 100000
        data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet_ontonet"
    elif args.level == 'ins':
        test_step = 20000
        data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet"
    elif args.level == '15k':
        data_path ="/home/skl/yl/FB15k"


    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if args.train:
        logset.set_logger(root_path,"init_train.log")
    else:
        logset.set_logger(root_path,'test.log')
    
    with_type_cons = True
    if args.train_type == 'rel_spl':
        dataset, ite,type2eids, eid2typeid = handle_things_for_rotate(data_path,args)
    else:
        # 读取数据集：
        if with_type_cons:
            dataset, ite = build_dataset_new_nagtive(data_path,args)
        else:
            dataset, ite = build_dataset(data_path,args)

    logging.info('Model: %s' % args.model)
    logging.info('On nentity: %s' % dataset.nentity)
    logging.info('On nrelatidataset. %s' % dataset.nrelation)
    logging.info('max step: %s' % max_step)
    logging.info('gamma: %s' % g_ons)
    logging.info('lr: %s' % lr)

    disjoint = read_disjonit("/home/skl/yl/data/YAGO3-1668k/yago_insnet/type2id.txt")
   
    # # model: TransE, RotatE, RotPro, DistMult, TuckER
    if(args.model == 'HAKE'):
        model = HAKE(dataset.nentity, dataset.nrelation, dim, g_ons, args.mode_weight, args.phase_weight)
    elif(args.model == 'RotPro'):
        model = RotPro(dataset.nentity,dataset.nrelation,dim,1.0,g_ons)
    elif(args.model == 'RotatE'):
        model = RotatE(dataset.nentity,dataset.nrelation,dim, gamma=g_ons)
    elif(args.model == 'HAKEpro'):
        model = HAKEpro(dataset.nentity,dataset.nrelation,dim,g_ons,args.mode_weight,args.phase_weight)
    elif(args.model == 'TransE'):
        model = TransE(dataset.nentity, dataset.nrelation, dim, gamma=g_ons)
    elif(args.model == 'BoxLevel'):
        model = BoxLevel(dataset.nentity, dataset.nrelation, dim)
    elif(args.model == 'SpereLevel'):
        model = SphereLevel(dataset.nentity, dataset.nrelation, dim,gamma=g_ons)
    elif(args.model =='PairE'):
        model = PairRE(dataset.nentity, dataset.nrelation, dim,gamma=g_ons)
    elif(args.model =='DTransE'):
        model = DTransE(dataset.nentity, dataset.nrelation, dim,gamma=g_ons)
    elif(args.model =='SemRotatE'):
        model = SemRotatE(dataset.nentity, dataset.nrelation, dim,type2eids, eid2typeid,gamma=g_ons)
    # model = DistMult(dataset.nentity, dataset.nrelation, dim, gamma=g_ons)

    if cuda:
        model = model.cuda()
        for key in disjoint.keys():
            disjoint[key] = torch.tensor(disjoint[key],dtype=torch.long).cuda()

    loss_function_on = NSSAL(g_ons,True,adversarial_temperature=args.adversial_temp)
    # loss_function_on = MRL(g_ons)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    # 如果有保存模型则，读取模型,进行测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']
        # save_embedding(model.save_model_embedding(),root_path)

    # 设置学习率更新策略
    # lr_scheduler = StepLR(optimizer, warm_up_steps, decay,verbose=False)
    # lr_scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)
    lr_scheduler = MultiStepLR(optimizer,milestones=[50000,120000], gamma=decay)
    # lr_scheduler = MultiStepLR(optimizer,milestones=[30000,70000,110000,160000], gamma=decay)
    logsB = []
    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":100000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }
    valid_inter, valid_outer = split_triples(dataset.valid,dataset.relation2id)

    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad(set_to_none=True)
            if step % 2 == 0:
                log2,loss_on = train_rel_split_step(ite['inter'],model,loss_function_on,cuda,args,train_type='inter')
            else:
                log2,loss_on = train_rel_split_step(ite['outer'],model,loss_function_on,cuda,args,train_type='outer')

            # if with_type_cons:
            #     log2,loss_on = train_with_disjoint_negative(ite['train'],model,loss_function_on,cuda,args,disjoint=disjoint)
            # else:
            #     log2,loss_on = train_step_old(ite['train'],model,loss_function_on,cuda,args,disjoint=disjoint)
            
            loss_on.backward()
            optimizer.step()
            lr_scheduler.step()
            logsB.append(log2)

            if step % log_steps == 0 :
                logging_log(step,logsB)
                logsB = []

            if step % test_step == 0  :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
                    "gamma":g_ons, "dim":dim,
                }

                logging.info('Valid inter at step: %d' % step)
                metrics = test_step_f(model, valid_inter, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse,train_type='inter')
                logset.log_metrics('Valid ',step, metrics)

                logging.info('Valid outer at step: %d' % step)
                metrics = test_step_f(model, valid_outer, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse,train_type='outer')
                logset.log_metrics('Valid ',step, metrics)
                ModelUtil.save_best_model(metrics,bestModel,model,optimizer,save_variable_list, args)

                
        valid_inter, valid_outer = split_triples(dataset.test,dataset.relation2id)

        logging.info('Valid inter at step: %d' % step)
        metrics = test_step_f(model, valid_inter, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse,train_type='inter')
        logset.log_metrics('Valid ',step, metrics)
        logging.info('Valid outer at step: %d' % step)
        metrics = test_step_f(model, valid_outer, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse,train_type='outer')
        logset.log_metrics('Valid ',step, metrics)

        # logging.info('Test at step: %d' % step)
        # metrics = test_step_f(model, dataset.test, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        # logset.log_metrics('Test ',step, metrics)
                
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
           "gamma":g_ons, "dim":dim,
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        if args.level =='15k':
            data_path ="/home/skl/yl/FB15k/mapping"
            test_file = ['121.txt','12n.txt','n21.txt','n2n.txt']
            for file_name in test_file:
                file = os.path.join(data_path,file_name)
                triples = DP.read_triples(file,entity2id=dataset.entity2id,relation2id=dataset.relation2id)
                logging.info('Test at patter: %s' % file_name[:-4])
                logging.info('predicate Head : %s' % file_name[:-4])
                metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='head')
                logset.log_metrics('Valid ',step, metrics)
                logging.info('predicate Tail : %s' % file_name[:-4])
                metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='tail')
                logset.log_metrics('Valid ',step, metrics)      

    # file_path = os.path.join(data_path,"relation","realtion_list.txt")
    # relations = []
    # with open(file_path) as fin:
    #     relations = fin.readlines()

    step = max_step       
    if args.test:
        # 实现bad case 分析相关的测试
        metrics,bad_case_list = test_print_bad_case(model, dataset.test, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda)
        # 暂时先将bad case 进行缓存，之后进行细致点的分析
        jsonAttr = json.dumps(bad_case_list,ensure_ascii=False)
        with open(os.path.join(root_path,'base_case.json'),'w',encoding='utf-8') as f:
            f.write(jsonAttr)
        f.close()




        # logging.info('Train Data at step: %d' % step)
        # metrics = test_step_f(model, dataset.train, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda)
        # logset.log_metrics('Train Data  ',step, metrics)
        # if args.level == '15k':
        #     data_path ="/home/skl/yl/FB15k/mapping"
        #     test_file = ['121.txt','12n.txt','n21.txt','n2n.txt']
        #     for file_name in test_file:
        #         file = os.path.join(data_path,file_name)
        #         triples = DP.read_triples(file,entity2id=dataset.entity2id,relation2id=dataset.relation2id)
        #         logging.info('Test at patter: %s' % file_name[:-4])
        #         logging.info('predicate Head : %s' % file_name[:-4])
        #         metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='head')
        #         logset.log_metrics('Valid ',step, metrics)
        #         logging.info('predicate Tail : %s' % file_name[:-4])
        #         metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='tail')
        #         logset.log_metrics('Valid ',step, metrics)
        # else:
        #     logging.info('Train at step: %d' % step)
        #     metrics = test_step_f(model, dataset.train, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        #     logset.log_metrics('Train ',step, metrics)

        #     # logging.info('Valid at step: %d' % step)
        #     # metrics = test_step_f(model, dataset.valid, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        #     # logset.log_metrics('Valid ',step, metrics)

        #     # logging.info('Test at step: %d' % step)
        #     # metrics = test_step_f(model, dataset.test, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        #     # logset.log_metrics('Test ',step, metrics)
        
        # rellist, rel2triples = DataTool.readSplitRelations(os.path.join(data_path,'split'))
        # for r in rellist:
        #     logging.info('Test at realtion: %s' % r)
        #     triples = rel2triples[r]
        #     metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda)
        #     logset.log_metrics('Test ',step, metrics)
        