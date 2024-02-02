from util.data_process import DataTool
from util.data_process import DataProcesser as DP
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
from core.TransC import TransC
from core.PairRE import PairRE
import torch.nn as nn
import yaml 
import json
import numpy as np
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator,OneShotIterator,OneToNDataset,SampleOne2NDataset,BoxTripleDataset
from loss import NSSAL,MRL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random

from core.UniBox import UniBox

def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)

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

    # if not onType is None:
    #     if onType == 'head':
    #         test_dataset_list = [test_dataloader_head]
    #     else:
    #         test_dataset_list = [test_dataloader_tail]
    # else:
    #     if not inverse:
    #         test_dataset_list = [test_dataloader_tail,test_dataloader_head]
    #     else:
    #         test_dataset_list = [test_dataloader_tail]
    test_dataset_list = [test_dataloader_head,test_dataloader_tail]
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
                    negative_score = model.predict(h,r, negative_sample,mode=mode)
                    positive_arg = t
                else:
                    negative_score = model.predict(negative_sample,r,t,mode=mode)
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
                    if ranking < 10: 
                        badCase = {}
                        badCase['mode'] =mode
                        badCase['true'] = (h[i].cpu().numpy().tolist(),r[i].cpu().numpy().tolist(),t[i].cpu().numpy().tolist())
                        badCase['top10'] = negative_sample[i][argsort[i,:10]].cpu().numpy().tolist()
                        badCase['Topscore'] = score[i][argsort[i,:10]].cpu().numpy().tolist()
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

def test_step_f(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None,head_num=None,tail_num=None,level=None):
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
                    negative_score = model.predict(h,r, negative_sample,mode=mode,level=level)
                    positive_arg = t
                else:
                    negative_score = model.predict(negative_sample,r,t,mode=mode,level=level)
                    positive_arg = h
             
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
    parser.add_argument("--label_smoothing", type=float, default=0.0)

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

    # 选择数据集
    parser.add_argument("--level",type=str,default='ins')
    parser.add_argument("--data_inverse",action='store_true')
    parser.add_argument("--configName",type=str,default='baseConfig')

    return parser.parse_args(args)

def save_embedding(emb_map,path,item=0):
    for key in emb_map.keys():
        np.save(
            os.path.join(path,key+str(item)),emb_map[key].detach().cpu().numpy()
        )

def build_sub_dataset(data_path,args,norm, reverse, trans_cloase = False,n_size = 200,head_num=None,tail_num=None):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    # reverse_train = on.get_reverse_train()
    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t',head_num,tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt',head_num,tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
    on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)

    # # 只使用了hr_all 数据进行训练
    # on_train_t = DataLoader(SampleOne2NDataset(on.train,reverse_train, on.nentity, on.nrelation,norm_init=norm,reverse_init=reverse),
    #         batch_size=args.batch_size,
    #         shuffle=True, 
    #         num_workers=max(1, 4//2),
    #         collate_fn=SampleOne2NDataset.collate_fn
    # )
    # on_train_iterator = OneShotIterator(on_train_t)
   
    return on,on_train_iterator

def build_ins_dataset(data_path,args,head_num=None, tail_num=None):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t',head_num,tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt',head_num,tail_num),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
  
    return on,on_train_iterator

def read_typeOf_data(data_path,args,entity2id,type2id,nentity, ntype):
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
    on_train_t = DataLoader(NagativeSampleDataset(train, nentity, 1, n_size, 'hr_t',nentity,ntype),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    # 
    if False:
        on_train_h = DataLoader(NagativeSampleDataset(train, nentity, 1, n_size, 'h_rt',nentity,ntype),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
   
    return dataset, on_train_iterator

def read_triples(file_path, h_dic,t_dic):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')  
            triples.append((h_dic[h],38,t_dic[t]))
    return triples  


def overwrite_args(args, baseConfig, modelConfig):
    args.batch_size = modelConfig['batch_size']
    args.test_step = baseConfig['valid_step']
    args.loss_weight = modelConfig['regu_weight']

if __name__=="__main__":
    # 读取4个数据集
    args = set_config()
    with open('./config/unibox.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]
    cuda = baseConfig['cuda']
    init_step  = 0
    save_steps = baseConfig['save_step']
    n_size = modelConfig['n_size']
    log_steps = 1000
    overwrite_args(args, baseConfig, modelConfig)
    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    args.save_path = root_path
    init_path = args.init
    max_step   = modelConfig['max_step']

    # lr = modelConfig['lr']
    decay = modelConfig['decay']

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if args.train:
        logset.set_logger(root_path,"train.log")
    else:
        logset.set_logger(root_path,'test.log')
        
    
    # todo：这里需要读取三个数据集
    instance_data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet/"
    typeOf_data_path = "/home/skl/yl/data/YAGO3-1668k/yago_type_new/"
    subOf_data_path = '/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/'
   
    subOf_dataset, subOf_ite = build_sub_dataset(subOf_data_path,args,modelConfig['norm_init'],modelConfig['reverse_init'],True, n_size=modelConfig['box_n_size'])
    instance_dataset, instance_ite = build_ins_dataset(instance_data_path,args)
    typeOf_dataset, typeOf_ite = read_typeOf_data(typeOf_data_path,args,instance_dataset.entity2id,subOf_dataset.entity2id,instance_dataset.nentity,subOf_dataset.nentity)


    g_instance  = modelConfig['g_ins']
    g_typeOf = modelConfig['g_type']
    g_subOf = modelConfig['g_subOf']


    logging.info('Model: UniBox')
    logging.info('Instance nentity: %s' % instance_dataset.nentity)
    logging.info('Instance nrelatidataset. %s' % instance_dataset.nrelation)
    logging.info('SubClass Of nentity. %s' % subOf_dataset.nentity)

    logging.info('max step: %s' % max_step)
    logging.info('Instance gamma: %s' % g_instance)
    logging.info('TypeOf gamma: %s' % g_typeOf)
    logging.info('SubClassOf gamma: %s' % g_subOf)
    logging.info('box lr: %s' %  modelConfig['box_lr'])
    logging.info('lr: %s' %  modelConfig['other_lr'])

    ent2rel = UniBox.build_ent2rel(instance_dataset.train,instance_dataset.nentity,instance_dataset.nrelation)
    ent2rel = ent2rel.detach()
    print(ent2rel.dtype)
    if cuda:
        ent2rel = ent2rel.cuda()
    model = UniBox(instance_dataset.nentity,instance_dataset.nrelation,subOf_dataset.nentity,
        type_dim = modelConfig['type_dim'],
        entity_dim=modelConfig['e_dim'],
        gamma=modelConfig['g_ins'],
        init_interval_center=0.5,
        box_reug=modelConfig['regu_weight'],
        ent2rel=ent2rel)
    if cuda:
        model = model.cuda()
        
    
    loss_function_instance = NSSAL(modelConfig['g_ins'],True,adversarial_temperature=modelConfig['adv_temp'])
    loss_function_typeOf = NSSAL(modelConfig['g_type'],True,adversarial_temperature=modelConfig['adv_temp'])
    loss_function_subOf = nn.BCELoss()
  
    typeEmb = [param for name,param in model.named_parameters() if name == 'type_model.init_tensor']
    # otherPara = [param for name,param in model.named_parameters() if name != 'type_model.init_tensor' and name != 'rel_trans.weight']
    otherPara = [param for name,param in model.named_parameters() if name != 'type_model.init_tensor']
    # typeOf = [param for name,param in model.named_parameters() if name == 'rel_trans.weight']
    optimizer = torch.optim.Adam([{
         'params':filter(lambda p: p.requires_grad , typeEmb), 
         'lr': modelConfig['box_lr']
        },
        {
         'params':filter(lambda p: p.requires_grad , otherPara), 
            'lr': modelConfig['other_lr']
        }
        # ,
        #  {
        #  'params':filter(lambda p: p.requires_grad , typeOf), 
        #     'lr': modelConfig['typeOf_lr']
        # }
        ],
        lr=modelConfig['other_lr']
    )
     # 需要答应一下优化的是哪些参数，检查是不是对的
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']
        logging.info('box_lr: %s' % optimizer.state_dict()['param_groups'][0]['lr'])
        logging.info('ins_lr: %s' % optimizer.state_dict()['param_groups'][1]['lr'])
        # logging.info('type_lr: %s' % optimizer.state_dict()['param_groups'][2]['lr'])
        save_embedding(model.save_model_embedding(),root_path)

    # 设置学习率更新策略
    lr_scheduler = MultiStepLR(optimizer,milestones=[15000,30000,50000,65000,80000], gamma=decay)
   
    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":1000000000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }

    loade_model_debug = False
    if loade_model_debug:
        print(model.value_of_index(0))
        value = []
        for i in range(subOf_dataset.nentity):
            if model.value_of_index(i) < -20:
                a = model.entity_embedding[i].Z - model.entity_embedding[i].z
                print(a)
            value.append(model.value_of_index(i).cpu().detach())

        value = np.array(value)
        print(np.mean(value))
        print(np.max(value))
        print(np.median(value))
        print(np.min(value))

    loss_log  = []

    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad(set_to_none=True)
            loss = 0
            log,loss = UniBox.train_step(model,optimizer,instance_ite,subOf_ite,typeOf_ite,
                loss_function_instance,loss_function_typeOf,loss_function_subOf,baseConfig['cuda'],step=step,reg_weight=modelConfig['regu_weight'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_log.append(log)
            if step % log_steps == 0 :
                logging_log(step,loss_log)
                loss_log = []
            
            # if step % 1000 == 0:         # 每训练1k个step 保存 Box 的 embedding
            #     save_embedding(model.save_model_embedding(),root_path,item=step)


            if step % 10000== 0:
                logging.info('Valid SubOf at step: %d' % step)
                metrics = test_step_f(model, subOf_dataset.valid, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda,modelConfig['data_reverse'],level='subClass')
                logset.log_metrics('Valid ',step, metrics)

            if step % args.test_step == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),'step':step
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)
                logging.info('Valid typeOf at step: %d' % step)
                metrics = test_step_f(model, typeOf_dataset['valid'], typeOf_dataset['all_true_triples'],0,1,cuda,inverse=True,level='typeOf',head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity)
                logset.log_metrics('Valid ',step, metrics)
                
                logging.info('Valid instance at step: %d' % step)
                metrics = test_step_f(model, instance_dataset.valid, instance_dataset.all_true_triples,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'],level='instance')
                logset.log_metrics('Valid ',step, metrics)

                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)

        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Test SubOf at step: %d' % step)
        metrics = test_step_f(model, subOf_dataset.test, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda,modelConfig['data_reverse'],level='subClass')
        logset.log_metrics('Test ',step, metrics)
        
        logging.info('Test typeOf at step: %d' % step)
        metrics = test_step_f(model, typeOf_dataset['test'], typeOf_dataset['all_true_triples'],0,1,cuda,inverse=True,level='typeOf',head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity)
        logset.log_metrics('Valid ',step, metrics)

        logging.info('Test instance at step: %d' % step)
        metrics = test_step_f(model, instance_dataset.test, instance_dataset.all_true_triples,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'],level='instance')
        logset.log_metrics('Test ',step, metrics)

    if args.test :
        # logging.info('Test SubOf at step: %d' % init_step)
        # metrics,bad_case_list = test_print_bad_case(model, subOf_dataset.test, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda,modelConfig['data_reverse'])
        # logset.log_metrics('Test ',init_step, metrics)

        logging.info('Test SubOf at step: %d' % init_step)
        metrics = test_step_f(model, subOf_dataset.test, subOf_dataset.all_true_triples,subOf_dataset.nentity,subOf_dataset.nrelation,cuda,modelConfig['data_reverse'],level='subClass')
        logset.log_metrics('Test ',init_step, metrics)
        
        logging.info('Test typeOf at step: %d' % init_step)
        metrics = test_step_f(model, typeOf_dataset['test'], typeOf_dataset['all_true_triples'],0,1,cuda,inverse=True,level='typeOf',head_num=instance_dataset.nentity,tail_num=subOf_dataset.nentity)
        logset.log_metrics('Valid ',init_step, metrics)

        logging.info('Test instance at step: %d' % init_step)
        metrics = test_step_f(model, instance_dataset.test, instance_dataset.all_true_triples,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'],level='instance')
        logset.log_metrics('Test ',init_step, metrics)

        # jsonAttr = json.dumps(bad_case_list,ensure_ascii=False)
        # with open(os.path.join(root_path,'base_case.json'),'w',encoding='utf-8') as f:
        #     f.write(jsonAttr)
        # f.close()