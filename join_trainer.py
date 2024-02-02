from collections import defaultdict
from curses import init_pair
from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
from core.TransER import TransER
import torch
from torch.optim.lr_scheduler import StepLR
from util.dataloader import NagativeSampleDataset,MultiShotItertor,OneShotIterator
from loss import NSSAL,MRL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random



def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)

def read_dic(file):
    with open(file,encoding='utf-8') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    return entity2id

def read_cross_data(data_path):
    ins_entity_dic = read_dic(os.path.join(data_path,"in_entity2id.dic"))
    on_entity_dic = read_dic(os.path.join(data_path,"on_entity2id.dic"))
    relation_dic = read_dic(os.path.join(data_path,"cross_relation2id.dic"))
    train = read_triples(os.path.join(data_path,"cross/train.txt"),ins_entity_dic,relation_dic,on_entity_dic)
    test = read_triples(os.path.join(data_path,"cross/test.txt"),ins_entity_dic,relation_dic,on_entity_dic)
    return train,test

def read_triples(file_path, h_dic,r_dic,t_dic):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')  
            triples.append((h_dic[h],r_dic[r],t_dic[t]))
    return triples  

def train_step(train_iterator,model,level,loss_function,cuda,optimizer,al,cg_detach=True):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
    
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
   
    negative_score = model(h,r, negative_sample, mode=mode,level=level,al=al,cg_detach=cg_detach)
    positive_score = model(h,r,t,level=level,al=al,cg_detach=cg_detach)

    # loss = loss_function(positive_score, negative_score,None)

    loss = loss_function(positive_score, negative_score)
    
    log = {
        level+'_loss': loss.item()
    }
    return log, loss
    
def test_step_f(model, test_triples, all_true_triples,nentity,nrelation,level,cuda=True,al="Self"):
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
        batch_size=128,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataset_list = [test_dataloader_tail]
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0
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
                    negative_score = model(h,r, negative_sample,level=level, mode=mode,al=al)
                else:
                    negative_score = model(negative_sample,r,t,level=level, mode=mode,al=al)
             
                score = negative_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)
                positive_arg = t
               
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

def read_model():
    pass


def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--test', action='store_true', help='use GPU')

    parser.add_argument('--ins', action='store_true', help='use GPU')
    parser.add_argument('--ons', action='store_true', help='use GPU')
    parser.add_argument('--JOIE', action='store_true', help='use GPU')
    parser.add_argument('--JOIE_P', action='store_true', help='use GPU')

    
    parser.add_argument('--max_step', type=int,default=200000, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)

    parser.add_argument("--ins_dim", type=int, default=200)
    parser.add_argument("--on_dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=40000)

    parser.add_argument("--loss_function", type=str)
    parser.add_argument("--g_ins",type=int,default=20)
    parser.add_argument("--g_on",type=int,default=23)
    parser.add_argument("--g_type",type=int,default=5)
    parser.add_argument("--g_level",type=int,default=5)

    parser.add_argument("--model",type=str)
    parser.add_argument("--rel_num",type=int)
    parser.add_argument("--init",type=str)

    return parser.parse_args(args)


def build_data_iterator(ins, on, cross_train, level):
    ins_train = DataLoader(NagativeSampleDataset(ins.train, ins.nentity, ins.nrelation, n_size, 'hr_t'),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NagativeSampleDataset.collate_fn
    )
    
    on_train = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    cross_train = DataLoader(
            NagativeSampleDataset(cross_train,on.nentity,1,n_size,'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn)
    level_train = DataLoader(
            NagativeSampleDataset(level.train,on.nentity,1,n_size,'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn)

    ins_iterator = OneShotIterator(ins_train)
    on_iterator = OneShotIterator(on_train)
    level_iterator = OneShotIterator(level_train)
    cross_iterator = OneShotIterator(cross_train)
    return  ins_iterator, on_iterator, level_iterator, cross_iterator


def one_hop_relation(triples,  rel_num,entity_num,relation_num):
    
    pad_id = relation_num

    entity_r = defaultdict(set)
    for h,r,t in triples:
        entity_r[h].add(r)
    unique_1hop_relations = [
        random.sample(entity_r[i], k=min(rel_num, len(entity_r[i]))) + [pad_id] * (rel_num-min(len(entity_r[i]), rel_num))
        for i in range(entity_num)
    ]
    return unique_1hop_relations

def updateEpoch(stepW, ins_iterator,on_iterator,level_iterator,cross_iterator,model,loss_function_in,cuda,optimizer,al,cg_detach):
    log1 = None
    log2 = None
    log3 = None 
    log4 = None
    if stepW == 0:
        log1,loss_in = train_step(ins_iterator,model,"in",loss_function_in,cuda,optimizer,al=al,cg_detach=cg_detach)
        log2,loss_on = train_step(on_iterator,model,"on",loss_function_on,cuda,optimizer,al=al,cg_detach=cg_detach)
        loss_in.backward()
        loss_on.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log3,loss_level = train_step(level_iterator,model,"level",loss_function_level,cuda,optimizer,al=al,cg_detach=cg_detach)
        loss_level.backward()
        optimizer.step()
    else:
        log4,loss_cross = train_step(cross_iterator,model,"cross",loss_function_type,cuda,optimizer,al=al,cg_detach=cg_detach)
        loss_cross.backward()
        optimizer.step()
    return log1, log2, log3, log4 

def updateAllEveryStep(ins_iterator,on_iterator,level_iterator,cross_iterator,model,loss_function_in,cuda,optimizer,al,cg_detach):
    log3 = None
    log1,loss_in = train_step(ins_iterator,model,"in",loss_function_in,cuda,optimizer,al=al,cg_detach=cg_detach)
    log2,loss_on = train_step(on_iterator,model,"on",loss_function_on,cuda,optimizer,al=al,cg_detach=cg_detach)
    # log3,loss_level = train_step(level_iterator,model,"level",loss_function_level,cuda,optimizer,al=al,cg_detach=cg_detach)
    log4,loss_cross = train_step(cross_iterator,model,"cross",loss_function_type,cuda,optimizer,al=al,cg_detach=cg_detach)
    loss_in.backward()
    loss_on.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss_cross = loss_cross * 1.2
    loss_cross.backward()
    optimizer.step()
    return log1, log2, log3, log4


if __name__=="__main__":
    # 读取4个数据集
    args = set_config()

    cuda = args.cuda
 
    init_step  = 0
    save_steps = 10000
    max_step   = args.max_step
   

    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    data_path = "/home/skl/yl/yago"
    init_path = args.init

    batch_size = args.batch_size
    test_step = args.test_step
    on_dim = args.on_dim
    in_dim = args.ins_dim

    lr = args.lr
    decay = args.decay
    warm_up_steps = args.warm_up_step

    n_size = 256
    log_steps = 1000

    TRAIN_INS = args.ins
    TRAIN_ONS = args.ons
    if  TRAIN_INS or TRAIN_ONS:
        TRAIN_JOIN = False
    else:
        TRAIN_JOIN = True

    TEST = args.test
    if TEST:
        root_path = os.path.join(root_path,"test")

    if args.JOIE:
        al = "JOIE"
    elif args.JOIE_P:
        al = "Self"
    else:
        al = "Else"

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    logset.set_logger(root_path)
    
    # 数据集构造
    ins = DP(os.path.join(data_path,"yago_insnet/"),idDict=True, reverse=False)
    on = DP(os.path.join(data_path,"yago_ontonet/"),idDict=True, reverse=False)
    on = DP(os.path.join(data_path,"yago_type//"),idDict=True, reverse=False)

    cross_train,cross_test = read_cross_data(data_path)
    if args.JOIE_P:
        level = DP(os.path.join(data_path,"level"),idDict=True, reverse=False)
    else:
        level = DP(os.path.join(data_path,"level"),idDict=True, reverse=True)

    ins_iterator,on_iterator, level_iterator,cross_iterator = build_data_iterator(ins, on, cross_train, level)

    # todo: 计算实体的关系的index_table, 本体和实例层分开表示
    ins_e2r = one_hop_relation(ins.all_true_triples,args.rel_num, ins.nentity,ins.nrelation)
    ons_e2r = one_hop_relation(on.all_true_triples,args.rel_num, on.nentity,on.nrelation)

    logging.info('On nentity: %s' % on.nentity)
    logging.info('On nrelation: %s' % on.nrelation)
    logging.info('In nentity: %s' % ins.nentity)
    logging.info('In nrelation: %s' % ins.nrelation)

    logging.info('AL: %s' % al)
    logging.info('Path: %s' % root_path)

    # 构造模型
    model = TransER(on.nentity,on.nrelation+1,on_dim,ins.nentity,ins.nrelation+1,in_dim,ins_e2r,ons_e2r)
    if cuda:
        model = model.cuda()
    # 设置优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    # 设置学习率更新策略
    # 如果有保存模型则，读取模型,进行测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']
    lr_scheduler = StepLR(optimizer, warm_up_steps, decay,verbose=False)
   
    if al == "JOIE":
        # loss_function_in = MRL(0.5)
        # loss_function_on = MRL(0.5)
        # loss_function_type = MRL(0.5)
        # loss_function_level = MRL(0.5)
        # 之前的设置
        # loss_function_in = NSSAL(20,False)
        # loss_function_on = NSSAL(23,False)
        # loss_function_type = NSSAL(10,False)
        # loss_function_level = NSSAL(15,False)
        g_ins = 23
        g_ons = 15
        g_cross =14
        g_level = 15
        loss_function_in = NSSAL(g_ins,False)
        loss_function_on = NSSAL(g_ons,False)
        loss_function_type = NSSAL(g_cross,False)
        loss_function_level = NSSAL(g_level,False)
    else:
        CT = False
        if CT:
            g_ins = 20
            g_ons = 25
            g_cross = 2
            g_level = 5
            loss_function_in = NSSAL(20,False)
            loss_function_on = NSSAL(25,False)
            loss_function_type = NSSAL(2,False)
            loss_function_level = NSSAL(5,False)
        else:
            g_ins = 23   # 上一次是25
            g_ons = 13
            g_cross = 5
            g_level = 5
            loss_function_in = NSSAL(g_ins,False)
            loss_function_on = NSSAL(g_ons,False)
            loss_function_type = NSSAL(g_cross,False)
            loss_function_level = NSSAL(g_level,False)
            # loss_function_in = MRL(0.5)
            # loss_function_on = MRL(0.5)
            # loss_function_type = MRL(0.5)
            # loss_function_level = MRL(0.5)
    
    logsA = []
    logsB = []
    logsC = []
    logsD = []
    logsAll = []
    cg_detach=True
    step1 = 0
    step2 = 0
    step3 = 300000
    epoch = ["ins","cross"]
    stepW = 0
    if not TEST :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad(set_to_none=True)
            if TRAIN_INS:
                log1,loss_in = train_step(ins_iterator,model,"in",loss_function_in,cuda,optimizer,al=al)
                loss_in.backward()
                optimizer.step()
            if TRAIN_ONS:
                log2,loss_on = train_step(on_iterator,model,"on",loss_function_on,cuda,optimizer,al=al)
                # log3,loss_level = train_step(level_iterator,model,"level", loss_function_level, cuda, optimizer,al=al)
                log3 = {"MRR":0}
                if True:
                    loss = loss_on 
                    loss.backward()
                else:
                    # loss = loss_on + loss_level
                    loss_on.backward()
                    optimizer.step()
                    loss_level.backward()
                optimizer.step()
            if TRAIN_JOIN:
                if step < step1:
                    log1,loss_in = train_step(ins_iterator,model,"in",loss_function_in,cuda,optimizer,al=al)
                    log2,loss_on = train_step(on_iterator,model,"on",loss_function_on,cuda,optimizer,al=al)
                    # log3,loss_level = train_step(level_iterator,model,"level",loss_function_level,cuda,optimizer,al=al)
                    loss_in.backward()
                    loss_on.backward()
                    optimizer.step()
                    # loss_level.backward()
                    # optimizer.step()
                elif step < step2:
                    log4,loss_cross = train_step(cross_iterator,model,"cross",loss_function_type,cuda,optimizer,al=al)
                    loss_cross = loss_cross*0.5
                    loss_cross.backward()
                    optimizer.step()
                elif step < step3:
                    if cg_detach:
                        root_path = os.path.join(root_path,"futune")
                        cg_detach = False
                        if not os.path.exists(root_path):
                            os.makedirs(root_path)
                    log1,log2,log3,log4 = updateAllEveryStep(ins_iterator,on_iterator,level_iterator,cross_iterator,model,loss_function_in,cuda,optimizer,al,cg_detach)
             #       log1,log2,log3,log4 = updateEpoch(stepW,ins_iterator,on_iterator,level_iterator,cross_iterator,model,loss_function_in,cuda,optimizer,al,cg_detach)


            lr_scheduler.step()

            if TRAIN_INS:
                logsA.append(log1)
            if TRAIN_ONS:
                logsB.append(log2)
                logsC.append(log3)
            if TRAIN_JOIN:
                if step < step1:
                    logsA.append(log1)
                    logsB.append(log2)
                    # logsC.append(log3)
                elif step < step2:
                    logsD.append(log4)
                elif step < step3:
                    logsA.append(log1)
                    logsB.append(log2)
                    logsD.append(log4)

            if step % log_steps == 0 :
                if TRAIN_INS:
                    logging_log(step,logsA)
                if TRAIN_ONS:
                    logging_log(step,logsB)
                    logging_log(step,logsC)
                if TRAIN_JOIN:
                    if step < step1:
                        logging_log(step,logsB)
                        # logging_log(step,logsC)
                        logging_log(step,logsA)
                    elif step < step2:
                        logging_log(step,logsD)
                    elif step < step3:
                        logging_log(step,logsA)
                        logging_log(step,logsB)
                        logging_log(step,logsD)
                # logging_log(step,logsAll)
                logsA = []
                logsB = []
                logsC = []
                logsD = []
                logsAll = []
            
            if step % save_steps == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
                    "g_ins":g_ins,"g_ons":g_ons, "g_cross":g_cross, "g_level":g_level,
                    "ins_dim":in_dim,"ons_dim":on_dim,"rel_num":args.rel_num
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path)

            if step % test_step == 0 and step !=0:
                if TRAIN_INS or TRAIN_JOIN:
                    logging.info('Test as step: %d' % step)
                    metrics =test_step_f(model, ins.test, ins.all_true_triples,ins.nentity,ins.nrelation,'in',cuda,al=al)
                    logset.log_metrics('Test Ins',step, metrics)
                if TRAIN_ONS or TRAIN_JOIN:
                    logging.info('Test On at step: %d' % step)
                    metrics = test_step_f(model, on.test, on.all_true_triples,on.nentity,on.nrelation,'on',cuda,al=al)
                    logset.log_metrics('Test On',step, metrics)
                    metrics = test_step_f(model, level.test, level.all_true_triples,on.nentity,1,'level',cuda,al=al)
                    logset.log_metrics('Test level',step, metrics)   
                if TRAIN_JOIN:
                    cross_all = cross_train + cross_test
                    logging.info('Test Cross at step: %d' % step)
                    metrics = test_step_f(model, cross_test, cross_all,on.nentity,1,'cross',cuda,al=al)
                    logset.log_metrics('Test Cross',step, metrics)  

        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
            "g_ins":g_ins,"g_ons":g_ons, "g_cross":g_cross, "g_level":g_level,
            "ins_dim":in_dim,"ons_dim":on_dim,"rel_num":args.rel_num
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path)

    step = max_step       
    if TEST:
        if TRAIN_INS or TRAIN_JOIN:
            logging.info('Test as step: %d' % step)
            metrics =test_step_f(model, ins.test, ins.all_true_triples,ins.nentity,ins.nrelation,'in',cuda,al=al)
            logset.log_metrics('Test Ins',step, metrics)
        if TRAIN_ONS or TRAIN_JOIN:
            logging.info('Test as step: %d' % step)
            metrics = test_step_f(model, on.test, on.all_true_triples,on.nentity,on.nrelation,'on',cuda,al=al)
            logset.log_metrics('Test On',step, metrics)   
            metrics = test_step_f(model, level.test, level.all_true_triples,on.nentity,1,'level',cuda,al=al)
            logset.log_metrics('Test level',step, metrics)   
        if TRAIN_JOIN:
            cross_all = cross_train + cross_test
            logging.info('Test as step: %d' % step)
            metrics = test_step_f(model, cross_test, cross_all,on.nentity,1,'cross',cuda,al=al)
            logset.log_metrics('Test Cross',step, metrics)  
