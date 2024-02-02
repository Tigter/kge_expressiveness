from collections import defaultdict
from curses import init_pair
from importlib.util import module_for_loader
from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
from core.HAKE import HAKE
from core.RotPro import RotPro
from core.RotatE import RotatE
from core.HAKEpro import HAKEpro
from core.BoxLevel import BoxLevel
import torch
from torch.optim.lr_scheduler import StepLR
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator,OneShotIterator
from loss import NSSAL,MRL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import yaml
import random


def read_triples(file_path, h_dic,t_dic):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')  
            triples.append((h_dic[h],0,t_dic[t]))
            # triples.append((t_dic[t],1,h_dic[h]))
    return triples  

# def read_triples(file_path, entitydict,relationdict,head_begin=0,tail_begin=0):
#     triples = []
#     with open(file_path, encoding='utf-8') as fin:
#         for line in fin:
#             h, r, t = line.strip().split('\t')  
#             triples.append((entitydict[h]-head_begin,relationdict[r],entitydict[t]-tail_begin))
#             # triples.append((t_dic[t],1,h_dic[h]))
#     return triples  

def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)


def train_step(train_iterator,model,level,loss_function,cuda,detach=False, args=None, head_begin=0,tail_begin=0):
   
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
    
    h = positive_sample[:,0]+head_begin
    r = positive_sample[:,1]
    t = positive_sample[:,2]+tail_begin
    type_trans = False
    if level == 'type':
        type_trans = True 
    
    if mode == 'hr_t':
        negative_sample += tail_begin
        negative_score = model(h,r, negative_sample,mode,type_trans=type_trans)
    elif mode =='h_rt':
        negative_sample += head_begin
        negative_score = model(negative_sample,r,t,mode,type_trans=type_trans)
    positive_score = model(h,r,t,'hrt',type_trans=type_trans)
    loss = loss_function(positive_score, negative_score)
    # reg = model.caculate_constarin(args.gamma_m,args.beta,args.alpha)
    reg = 0
    log = {
        level+'_loss': loss.item()
    }
    return log, loss+reg

def train_mul_step(train_iterator,model,level,loss_function,cuda,optimizer,args,al='Else',cg_detach=True,loss_m=None,loss_p=None):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
    
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
   
    negative_score = model(h,r, negative_sample,mode,model_level='phase')
    positive_score = model(h,r,t,model_level='phase')

    loss1 = loss_p(positive_score, negative_score)
    loss1.backward()
    
    negative_score = model(h,r, negative_sample,mode,model_level='module')
    positive_score = model(h,r,t,model_level='module')
    loss2 = loss_m(positive_score, negative_score)

    loss2.backward()
    optimizer.step()

    log = {
        level+'_p_loss': loss1.item(),
        level+'_m_loss': loss2.item(),
    }
    return log, loss1+loss2
    
def test_step_f(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None,head_num=None,tail_num=None,level=None,head_begin=0,tail_begin=0):
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
            'hr_t',
            head_num,
            tail_num
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
            'h_rt',
            head_num,
            tail_num
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    if not inverse:
        test_dataset_list = [test_dataloader_tail, test_dataloader_head]
    else:
        test_dataset_list = [test_dataloader_tail]
    
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0
    # print(total_steps)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                batch_size = positive_sample.size(0)
                if cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()
                    
                h = positive_sample[:,0] + head_begin
                r = positive_sample[:,1]
                t = positive_sample[:,2] + tail_begin
                if mode == 'hr_t':
                    negative_sample += tail_begin
                    negative_score = model(h,r, negative_sample,mode=mode)
                    positive_arg = t - tail_begin
                elif mode =='h_rt':
                    negative_sample += head_begin
                    negative_score = model(negative_sample,r,t,mode=mode)
                    positive_arg = h - head_begin
             
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

def build_dataset(data_path,args,entity2id,type2id,head_num,tail_num):
    train = read_triples(os.path.join(data_path,"train.txt"),entity2id,type2id)
    valid = read_triples(os.path.join(data_path,"valid.txt"),entity2id,type2id)
    test = read_triples(os.path.join(data_path,"test.txt"),entity2id,type2id)
    all_true_triples = list(set(train+valid+test))
    dataset = {
        'train':train,
        'valid':valid,
        'test':test,
        'all_true_triples':all_true_triples
    }
    on_train_t = DataLoader(NagativeSampleDataset(train, head_num, 1, n_size, 'hr_t',
        head_num=head_num,tail_num=tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    # 
    if True:
        on_train_h = DataLoader(NagativeSampleDataset(train, head_num, 1, n_size, 'h_rt',
            head_num=head_num,tail_num=tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
   
    return dataset, on_train_iterator

def simple_dataset(on,batch_size):
    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_iterator = BidirectionalOneShotIterator(on_train_t,on_train_h)   
    return on_iterator 

def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--test', action='store_true', help='use GPU')

    parser.add_argument('--max_step', type=int,default=200000, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)

    parser.add_argument("--gamma", type=int, default=20)
    parser.add_argument("--on_dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=10000)

    parser.add_argument("--loss_function", type=str)
    parser.add_argument("--mode_weight",type=float,default=0.5)
    parser.add_argument("--phase_weight",type=float,default=0.5)
    parser.add_argument("--g_type",type=int,default=5)
    parser.add_argument("--g_level",type=int,default=5)

    parser.add_argument("--model",type=str)
    parser.add_argument("--init",type=str)

    parser.add_argument("--g_mode",type=float,default=0.5)
    parser.add_argument("--g_phase",type=float,default=0.5)

    parser.add_argument("--gamma_m",type=float,default=0.000001)
    parser.add_argument("--configName",type=str)
    parser.add_argument("--alpha",type=float,default=0.0005)
    parser.add_argument("--beta",type=float,default=1.5)

    return parser.parse_args(args)




if __name__=="__main__":
    # 读取4个数据集
    args = set_config()
    with open('./config/typeInfo.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]
    cuda = baseConfig['cuda']
 
    init_step  = 0
    log_steps = 1000
    save_steps =10000
    max_step   = modelConfig['max_step']

    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    # data_path = "/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet"
    # data_path = "/home/skl/yl/data/YAGO3-1668k/yago_type_new"
    data_path = "/home/skl/yl/data/YAGO3-1668k/yago_all_new"
    init_path = args.init

    args.batch_size = baseConfig['batch_size']
    test_step = baseConfig['valid_step']
    dim = modelConfig['e_dim']
   
    lr =  modelConfig['lr']
    decay = modelConfig['decay']
    warm_up_steps =  modelConfig['decay_step']

    n_size = 1000
    g_ons = modelConfig['g_ins']
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not args.test:
        logset.set_logger(root_path)
    else:
        logset.set_logger(root_path,'test.log')
    
    # 数据集构造
    on = DP(data_path,idDict=True, reverse=False)

    # all_test = read_triples(os.path.join(data_path,'all.txt'),on.entity2id,on.entity2id)
    # all_test = on.all_true_triples
    # on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t'),
    #         batch_size=batch_size,
    #         shuffle=True, 
    #         num_workers=max(1, 4//2),
    #         collate_fn=NagativeSampleDataset.collate_fn
    # )
    # on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt'),
    #         batch_size=batch_size,
    #         shuffle=True, 
    #         num_workers=max(1, 4//2),
    #         collate_fn=NagativeSampleDataset.collate_fn
    # )
    # on_iterator = BidirectionalOneShotIterator(on_train_t,on_train_h)

    instance_data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet/"
    typeOf_data_path = "/home/skl/yl/data/YAGO3-1668k/yago_type_new/"
    subOf_data_path = '/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/'

    subOf_dataset = DP(subOf_data_path,idDict=True, reverse=False)
    instance_dataset = DP(instance_data_path,idDict=True, reverse=False)

    sub_ite =simple_dataset(subOf_dataset,args.batch_size)
    ins_ite = simple_dataset(instance_dataset,args.batch_size)
    typeOf_dataset, on_iterator = build_dataset(typeOf_data_path,args,instance_dataset.entity2id,subOf_dataset.entity2id,
        instance_dataset.nentity,subOf_dataset.nentity,
       )

    loss_function_on = NSSAL(modelConfig['g_ins'])
    loss_function_typeOf = NSSAL(modelConfig['g_type'])
    loss_function_subOf = NSSAL(modelConfig['g_ons'])
    total_entity =  subOf_dataset.nentity + instance_dataset.nentity
    logging.info('On nentity: %s' % total_entity)
    logging.info('On nrelation: %s' % on.nrelation)
    logging.info('max step: %s' % max_step)
    logging.info('gamma: %s' % g_ons)
    logging.info('lr: %s' % lr)
   
    # model = HAKE(on.nentity,on.nrelation,on_dim,g_ons,args.mode_weight,args.phase_weight)
    model = RotPro(total_entity,on.nrelation,dim,1.0,g_ons)
    # model = RotatE(total_entity,on.nrelation,dim,g_ons)
    # model = HAKEpro(on.nentity,on.nrelation,on_dim,g_ons, args.mode_weight, args.phase_weight)
    # model = BoxLevel(on.nentity, on.nrelation, dim)
    
    if cuda:
        model = model.cuda()
    # 设置优化器
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
        print("init_step: %d" % init_step)
    
    # 设置学习率更新策略
    lr_scheduler = StepLR(optimizer, warm_up_steps, decay,verbose=False)
    logsA = []
    logsB = []
    logsC = []
    stepW = 0
    firs = True
    bestModel = {
        "MRR":0,
        "MR":1000000000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }
    if not args.test :
        print("start training......................")
        for step in range(init_step, max_step):
            optimizer.zero_grad(set_to_none=True)
            log1,loss_sub = train_step(sub_ite, model,"sub", loss_function_on, cuda, detach=True,args=args
                    ,head_begin=instance_dataset.nentity,tail_begin=instance_dataset.nentity)
            log2,loss_ins = train_step(ins_ite, model,"ins", loss_function_on, cuda, detach=True,args=args)
            if step > 60000:
                log3,loss_type = train_step(on_iterator, model,"type", loss_function_on, cuda, detach=True,args=args
                    ,head_begin=0,tail_begin=instance_dataset.nentity)
                logsC.append(log3)
                loss = loss_type+ loss_sub + loss_ins
            else:
                loss = loss_sub + loss_ins
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            logsA.append(log1)
            logsB.append(log2)
            
            if step % log_steps == 0 :
                logging_log(step,logsA)
                logsA = []

                logging_log(step,logsB)
                logsB = []
                if step > 60000:
                    logging_log(step,logsC)
                    logsC = []

            if step % save_steps == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
                    "g_ons":g_ons, "ons_dim":dim
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path)

            if step % 40000 == 0:
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
                "g_ons":g_ons, "ons_dim":dim,
                }
                logging.info('Valid Model  On at step: %d' % step)
                # metrics = test_step_f(model, typeOf_dataset['valid'], all_test,on.nentity,on.nrelation,cuda=cuda,inverse=False,head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity)
                metrics = test_step_f(model, subOf_dataset.valid, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda=cuda,inverse=False,
                    head_num=subOf_dataset.nentity,tail_num=subOf_dataset.nentity,
                    tail_begin=instance_dataset.nentity,head_begin=instance_dataset.nentity)
                logset.log_metrics('valid subOf',step, metrics)
                metrics = test_step_f(model, typeOf_dataset['valid'], typeOf_dataset['all_true_triples'],on.nentity,on.nrelation,cuda=cuda,inverse=False,
                    head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity,
                    head_begin=0,tail_begin=instance_dataset.nentity
                    )
                logset.log_metrics('valid typeOf',step, metrics)
                metrics = test_step_f(model, instance_dataset.valid, instance_dataset.all_true_triples,on.nentity,on.nrelation,cuda=cuda,inverse=False,
                    head_num=instance_dataset.nentity,tail_num=instance_dataset.nentity,
                    tail_begin=0,head_begin=0              
                    )
                logset.log_metrics('valid isntance',step, metrics)

                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path)

        metrics = test_step_f(model, subOf_dataset.valid, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda=cuda,inverse=False,
                head_num=subOf_dataset.nentity,tail_num=subOf_dataset.nentity,
                tail_begin=instance_dataset.nentity,head_begin=instance_dataset.nentity)
        logset.log_metrics('Test subOf',step, metrics)
        metrics = test_step_f(model, typeOf_dataset['test'], on.all_true_triples,on.nentity,on.nrelation,cuda=cuda,inverse=False,
            head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity,
            tail_begin=0,head_begin=instance_dataset.nentity
            )
        logset.log_metrics('Test typeOf',step, metrics)
        metrics = test_step_f(model, instance_dataset.test, instance_dataset.all_true_triples,on.nentity,on.nrelation,cuda=cuda,inverse=False,
            head_num=instance_dataset.nentity,tail_num=instance_dataset.nentity,
            tail_begin=0,head_begin=0              
            )

        logset.log_metrics('Test instance',step, metrics)
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
           "g_ons":g_ons, "ons_dim":dim,
        }

    step = max_step       
    if args.test:
        logging.info('Test On at step: %d' % step)
        metrics = test_step_f(model, on.test, on.all_true_triples,on.nentity,on.nrelation,'on',cuda)
        logset.log_metrics('Test On',step, metrics)

