# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:34:10 2024

@author: admin
"""
from arguments import args_parser
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,accuracy_score
import sys, torch, math, os, time
from models.Nets import LinearAttackModel
from utils.dataset_helper import ImageDataset 
from ML_utils import DNNTrain, DNNTest
import torch.nn.functional as F
from Privacy_utils import privacy_protect


def combination(n, k):
    """计算组合数 C(n, k)"""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
 
def permutation(n, k):
    """计算排列数 P(n, k)"""
    return math.factorial(n) // math.factorial(n - k)


def MIA_logRead(SV_args):
    shadowDataset, QueryDataset, inMemberDataset = [],[],[]
    querySampleSVin = dict()
    querySampleSVin_ref = dict()
    querySampleSVout = dict()
    querySampleSVout_ref = dict()
    queryValue = dict()
    queryValue_ref = dict()
    querySampleExcludeList = dict()
    replace_list = ['_recovery',
                    'withDP%s'%SV_args.privacy_protection_level,
                    'withQT%s'%SV_args.privacy_protection_level,
                    'withDR%s'%SV_args.privacy_protection_level,
                    '_seed10','_seed1']
    log_file = SV_args.log_file
    for replace_str in replace_list:
        log_file = log_file.replace(replace_str,"")
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = file.readlines()
            queryLabel = dict()
            resultantSV = []
            SV_var = dict()
            Current_SV = []
            exclude_list = []
            for line in lines:
                if 'shadowDataset:' in line:
                    shadowDataset = eval(line.split('shadowDataset:')[-1].strip())
                if 'QueryDataset:' in line:
                    QueryDataset = eval(line.split('QueryDataset:')[-1].strip())
                if 'inMemberDataset:' in line:
                    inMemberDataset = eval(line.split('inMemberDataset:')[-1].strip())
                
                if 'Current SV' in line:
                    Current_SV.append(eval(line.split('Current SV:')[-1]))
                if 'Final Resultant SVs:' in line:
                    resultantSV.append(
                        eval(line.split('Final Resultant SVs:')[-1]))
                    
                    SV_var[len(resultantSV)-1] = dict([
                        (player_idx, \
                         (resultantSV[-1][player_idx],
                          np.var([sv[player_idx] for sv in Current_SV]))) \
                        for player_idx in resultantSV[-1].keys()
                        ])
                    Current_SV=[]
                elif 'trn data idxs:' in line:
                    #print(",".join(map(str,sorted(shadow_dataset))))
                    shadow_dataset = eval(line.split('trn data idxs:')[-1].strip())
                    if sorted(shadow_dataset) not in exclude_list and\
                    shadow_dataset[-1]==shadow_dataset[-2]:
                        exclude_list.append(sorted(shadow_dataset))
                    
                elif 'SV_out' in line:
                    sample_idx = int(line.split("/")[0].split(" ")[-1])
                    SV_out = eval(line.split("SV_out")[-1].split(",")[0])
                    SV_in = eval(line.split(":")[-1])
                    
                    if sample_idx in querySampleSVout.keys():
                        querySampleSVout[sample_idx].append(SV_out)
                    else:
                        querySampleSVout[sample_idx] = [SV_out]
                    if sample_idx in querySampleSVout_ref.keys():
                        querySampleSVout_ref[sample_idx][
                            len(querySampleSVout[sample_idx])-1]=SV_var[0]
                    else:
                        querySampleSVout_ref[sample_idx]={
                            len(querySampleSVout[sample_idx])-1:SV_var[0]
                            }
                    
                    if sample_idx in querySampleSVin.keys():
                        querySampleSVin[sample_idx].append(SV_in)
                    else:
                        querySampleSVin[sample_idx] = [SV_in]
                    if sample_idx in querySampleSVin_ref.keys():
                        querySampleSVin_ref[sample_idx][
                            len(querySampleSVout[sample_idx])-1]=SV_var[1]
                            
                    else:
                        querySampleSVin_ref[sample_idx]= {
                            len(querySampleSVout[sample_idx])-1:SV_var[1]
                            }
                    resultantSV = []
                    SV_var = dict()
                elif "result:" in line:
                    results = line.split("result:")[-1].strip().split(" ")
                    queryValue[sample_idx] = eval(results[0])
                    queryLabel[sample_idx] = eval(results[1])
                    queryValue_ref[sample_idx] = SV_var[0]
                    resultantSV = []
                    SV_var = dict()
                    tmp = []
                    for shadow_dataset in exclude_list:
                        discard_member = [member for member in shadow_dataset \
                                          if shadow_dataset.count(member)>1]
                        shadow_dataset = set(shadow_dataset)
                        for member in discard_member:
                            shadow_dataset.discard(member)
                        sampledShadowDataset = ",".join(map(str,sorted(shadow_dataset)))
                        if sampledShadowDataset not in tmp:
                            tmp.append(sampledShadowDataset)
                    querySampleExcludeList[sample_idx] = tmp
                    exclude_list = []
                    
            for sample_idx in querySampleSVin.keys():
                if sample_idx not in queryValue.keys():
                    tmp = []
                    for shadow_dataset in exclude_list:
                        discard_member = [member for member in shadow_dataset \
                                          if shadow_dataset.count(member)>1]
                        shadow_dataset = set(shadow_dataset)
                        for member in discard_member:
                            shadow_dataset.discard(member)
                        sampledShadowDataset = ",".join(map(str,sorted(shadow_dataset)))
                        if sampledShadowDataset not in tmp:
                            tmp.append(sampledShadowDataset)
                    querySampleExcludeList[sample_idx] = tmp
                    exclude_list = []
                    
    return shadowDataset, QueryDataset, inMemberDataset,\
        queryValue, queryValue_ref, \
        querySampleSVin, querySampleSVin_ref,\
        querySampleSVout, querySampleSVout_ref, querySampleExcludeList
        
def MIA_addPrivacyProtection(qidx, value, ref, 
                             SV_in, SV_in_ref, 
                             SV_out, SV_out_ref, SV_args):
    # add privacy protection
    sortResult_change = []
    print('(bef PPM) sample %s\'s SV:'%qidx, SV_in, SV_out, value)
    
    for sv_idx in range(len(SV_out)):
        sortResult_bef = np.argsort(
            list([sv for (sv, var) in SV_out_ref[sv_idx].values()])).tolist()
        sortResult_bef = [sortResult_bef.index(k) \
                          for k in SV_out_ref[sv_idx].keys()]
        
        SV_with_protect = privacy_protect(
            SV_args.privacy_protection_measure, 
            SV_args.privacy_protection_level,
            dict([(k, sv) for (k,(sv, var)) in SV_out_ref[sv_idx].items()]),
            dict([(k, var) for (k,(sv, var)) in SV_out_ref[sv_idx].items()]))
        SV_out[sv_idx] = SV_with_protect[len(SV_with_protect)-1]
    
        sortResult_aft = np.argsort(
            list(SV_with_protect.values())).tolist()
        sortResult_aft = [sortResult_aft.index(k) \
                          for k in SV_with_protect.keys()]
        sortResult_change.append(
            np.mean(
                np.abs(np.array(sortResult_aft)-\
                       np.array(sortResult_bef))/\
                    len(SV_with_protect)
                    )
                )
            
    for sv_idx in range(len(SV_in)):
        sortResult_bef = np.argsort(
            list([sv for (sv, var) in SV_in_ref[sv_idx].values()])).tolist()
        sortResult_bef = [sortResult_bef.index(k) \
                          for k in SV_in_ref[sv_idx].keys()]
        
        SV_with_protect = privacy_protect(
            SV_args.privacy_protection_measure, 
            SV_args.privacy_protection_level,
            dict([(k, sv) for (k,(sv, var)) in SV_in_ref[sv_idx].items()]),
            dict([(k, var) for (k,(sv, var)) in SV_in_ref[sv_idx].items()]))#querySampleSVin_ref[qidx][sv_idx])
        SV_in[sv_idx] = SV_with_protect[len(SV_with_protect)-1]
        
        sortResult_aft = np.argsort(
            list(SV_with_protect.values())).tolist()
        sortResult_aft = [sortResult_aft.index(k) \
                          for k in SV_with_protect.keys()]
        sortResult_change.append(
            np.mean(
                np.abs(np.array(sortResult_aft)-\
                       np.array(sortResult_bef))/\
                    len(SV_with_protect)
                    )
                )
    
    sortResult_bef = np.argsort(
        list([sv for (sv, var) in ref.values()])).tolist()
    sortResult_bef = [sortResult_bef.index(k) \
                      for k in ref.keys()]
    
    SV_with_protect = privacy_protect(
        SV_args.privacy_protection_measure, 
        SV_args.privacy_protection_level,
        dict([(k, sv) for (k,(sv, var)) in ref.items()]),
        dict([(k, var) for (k,(sv, var)) in ref.items()]))#queryValue_ref[qidx])
    value = SV_with_protect[len(SV_with_protect)-1]
    
    sortResult_aft = np.argsort(
        list(SV_with_protect.values())).tolist()
    sortResult_aft = [sortResult_aft.index(k) \
                      for k in SV_with_protect.keys()]
    sortResult_change.append(
        np.mean(
            np.abs(np.array(sortResult_aft)-\
                   np.array(sortResult_bef))/\
                len(SV_with_protect)
                )
        #abs(sortResult_aft[len(SV_with_protect)-1]-\
        #    sortResult_bef[len(SV_with_protect)-1])/\
        #    len(SV_with_protect)
            )
    print('(aft PPM) sample %s\'s SV:'%qidx, SV_in, SV_out, value)   
    sortResult_change = np.mean(sortResult_change)
    print('sortResult_change for query sample%s:'%qidx, sortResult_change)
    return value, SV_in, SV_out, sortResult_change


def SV_compute(task, sampleIdx, datasetIdx, SV_args):
    targetedPlayer_idx = len(datasetIdx)
    task.players.idxs = datasetIdx + [sampleIdx]
    print('trn data idxs:', task.players.idxs)
    sys.stdout.flush()
    SV = task.run()
    return SV[targetedPlayer_idx], \
           dict([(pidx, (SV[pidx], np.var(task.SV_var[pidx])))\
                 for pidx in SV.keys()]) 

def MIA(maxIter, SV_args):
    '''
    The threat model we consider here is that 
    the attacker can compute the value of any data point among any datasets 
    (analogue to the setting of MIA against ML models where 
     the attacker can train models on any datasets he/she constructs).
    '''
    '''
    1. For each dataset we experiment on, we select 200 data points (members) 
    as the private dataset held by the central server. 
    2. We pick another 200 data points which serve as non-members. 
    3. Moreover, we leverage another 400 data points where 
    we can subsample “shadow dataset” in Algorithm 1. 
    4. We set the number of shadow datasets we sample as 32.
    5. The membership probability of every target sample 
       is determined by in_probability/out_probability;
    6. Adopting different thresholds can produce different confusion matrices 
       with different true positive rates and false positive rates, 
       which produce a ROC curve and its associated AUROC.
    '''
    #SV_args.scannedIter_maxNum=50
    shadowDataset, QueryDataset, inMemberDataset,\
    queryValue, queryValue_ref, \
    querySampleSVin, querySampleSVin_ref,\
    querySampleSVout, querySampleSVout_ref,\
    querySampleExcludeList = MIA_logRead(SV_args)
    
    
    task = Task(SV_args)  # DV-attack task
    dataIdx = list(task.players.idxs)
    if len(inMemberDataset)<=0:
        QueryDataset = np.random.choice(
            dataIdx, 8, replace=False).tolist()
        shadowDataset = list(set(dataIdx)-set(QueryDataset))
        inMemberDataset = np.random.choice(
            QueryDataset, int(len(QueryDataset)/2), replace=False).tolist()
        if len(inMemberDataset) < len(dataIdx)/2:
            inMemberDataset += np.random.choice(
                shadowDataset, int(len(dataIdx)/2-len(inMemberDataset)), 
                replace=False).tolist()
        print('shadowDataset: ',shadowDataset)
        print('QueryDataset: ',QueryDataset)
        print('inMemberDataset: ',inMemberDataset)
    
    if len(QueryDataset)>10:
        shadowDataset += QueryDataset[10:]
        QueryDataset = QueryDataset[:10]
        
    queryResults = []
    queryLabels = []
    sortChange = []
    numSamples_in_Shadow = max(int(len(shadowDataset)/4), 
                               int(len(shadowDataset)/maxIter))
    for qidx, z in enumerate(QueryDataset):
        
        SV_in = querySampleSVin[qidx] if qidx in querySampleSVin.keys() else []
        SV_in_ref = querySampleSVin_ref[qidx] if qidx in querySampleSVin_ref.keys() else []
        SV_out = querySampleSVout[qidx] if qidx in querySampleSVout.keys() else []
        SV_out_ref = querySampleSVout_ref[qidx] if qidx in querySampleSVout_ref.keys() else []
        sampledShadowDataset = shadowDataset[:numSamples_in_Shadow]
        exclude_list = (querySampleExcludeList[qidx] \
                        if qidx in querySampleExcludeList.keys() else [])
        print('Read results for query sample %s maxIter %s:'%(qidx, maxIter),
              len(SV_in), len(SV_in_ref),
              len(SV_out), len(SV_out_ref), len(exclude_list))
        #print(exclude_list, ",".join(map(str,sorted(sampledShadowDataset))),
        #      ",".join(map(str,sorted(sampledShadowDataset))) in exclude_list)
        for iter_ in range(len(SV_in), maxIter):
            while ",".join(map(str,sorted(sampledShadowDataset))) in exclude_list:
                sampledShadowDataset = np.random.choice(
                    shadowDataset, numSamples_in_Shadow, 
                    replace=False).tolist()
            exclude_list.append(",".join(map(str,sorted(sampledShadowDataset))))
            
            value, ref = SV_compute(task, z, sampledShadowDataset, SV_args)
            SV_out.append(value)
            SV_out_ref.append(ref)
            sampledShadowDataset.append(z)
            value, ref = SV_compute(task, z, sampledShadowDataset, SV_args)
            SV_in.append(value)
            SV_in_ref.append(ref)
            sampledShadowDataset = sampledShadowDataset[:-1]
            print('Query sample %s/%s Iter %s/%s: SV_out %s, SV_in: %s'%(
                qidx, len(QueryDataset), iter_, maxIter,
                SV_out[-1], SV_in[-1]))
            if len(exclude_list) >= combination(
                    len(shadowDataset), numSamples_in_Shadow):
                break
        print('Query sample %s/%s:'%(qidx, len(QueryDataset)),
              SV_out, SV_in)
           
        (value, ref) = (
            (queryValue[qidx], queryValue_ref[qidx])\
                if qidx in queryValue.keys() else\
                 SV_compute(task, z, inMemberDataset, SV_args))
        
        if SV_args.privacy_protection_measure != None:
            value, SV_in, SV_out, sc = MIA_addPrivacyProtection(
                qidx, value, ref, 
                SV_in, SV_in_ref,
                SV_out, SV_out_ref, SV_args)
            sortChange.append(sc)
        else:
            sortChange.append(0)
            
        in_mean = np.mean(SV_in)
        out_mean = np.mean(SV_out)
        in_std = np.std(SV_in)
        if in_std ==0:
            in_std = 10**(-20)
        out_std = np.std(SV_out)
        if out_std ==0:
            out_std = 10**(-20)
        
        in_probability = norm.pdf(value, loc=in_mean, scale=in_std)
        out_probability = norm.pdf(value, loc=out_mean, scale=out_std)
        if out_probability ==0:
            out_probability = 10**(-20)
        queryResults.append(in_probability/out_probability)
        queryLabels.append(int(z in inMemberDataset))
        print('Query sample %s/%s result: '%(qidx, len(QueryDataset)),
              value, queryLabels[-1], queryResults[-1], 
              in_probability, out_probability)
        
    print('queryLabels:',queryLabels, '\n'
          'queryResults:',queryResults,
          'sortChange', np.mean(sortChange))
    try:
        AUROC = roc_auc_score(queryLabels, queryResults)
    except Exception as e :
        print(e)
        AUROC = 0.5
    print('MIA AUROC: ', AUROC)
    ACC = accuracy_score(
        queryLabels, [int(r>1) for r in queryResults])
    print('MIA ACC: ', ACC)
    return AUROC

def FIA_logRead(SV_args):
    replace_list = ['_recovery',
                    'withDP%s'%SV_args.privacy_protection_level,
                    'withQT%s'%SV_args.privacy_protection_level,
                    'withDR%s'%SV_args.privacy_protection_level,
                    '_seed10','_seed1']
    log_file = SV_args.log_file
    for replace_str in replace_list:
        log_file = log_file.replace(replace_str,"")
    if not os.path.exists(log_file) or log_file == SV_args.log_file:
        log_file = log_file.replace('FIA',"FA")
        
    testSampleFeatureSV = dict()
    SV_var = dict()
    Current_SV = []
    randomTestData = dict() 
    auxiliary_index = []
    data = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = list(file.readlines())
            for no, line in enumerate(lines):
                if 'auxiliary_index' in line:
                    auxiliary_index = eval(
                        line.split('auxiliary_index')[-1].strip())
                elif 'test sample data:' in line:
                    data = line.split(
                        'test sample data:  tensor(')[-1].strip()
                    tmp = 1
                    while ')' not in data:
                        data += lines[no+tmp].strip()
                        tmp += 1
                    data = eval(data.replace(')',''))
                    
                if 'Current SV' in line:
                    Current_SV.append(eval(line.split('Current SV:')[-1]))
                if 'SV of test sample' in line:
                    sample_idx=int(line.split("/")[0].split(" ")[-1])
                    randomTestData[sample_idx] = np.array(data)
                    data = []
                    testSampleFeatureSV[sample_idx] = eval("{"+line.split("{")[-1])
                    SV_var[sample_idx] = dict([
                        (feature_idx, \
                        np.var([sv[feature_idx] for sv in Current_SV])) \
                        for feature_idx in testSampleFeatureSV[sample_idx].keys()
                        ])
                    Current_SV=[]
    
    print('read existing testSampleFeatureSV...')
    return randomTestData, auxiliary_index, testSampleFeatureSV, SV_var

def FIA_addPrivacyProtection(SV_args, testSampleFeatureSV, SV_var):
    sortResult_change = []
    for test_idx in testSampleFeatureSV.keys():
        print('(bef PPM) sample %s\'s SV:'%test_idx, 
              testSampleFeatureSV[test_idx])
        sortResult_bef = np.argsort(
            list(testSampleFeatureSV[test_idx].values())).tolist()
        sortResult_bef = [sortResult_bef.index(k) \
                          for k in testSampleFeatureSV[test_idx].keys()]
        
        testSampleFeatureSV[test_idx] = privacy_protect(
            SV_args.privacy_protection_measure, 
            SV_args.privacy_protection_level,
            testSampleFeatureSV[test_idx], 
            SV_var[test_idx])
        print('(aft PPM) sample %s\'s SV:'%test_idx, 
              testSampleFeatureSV[test_idx])
        sortResult_aft = np.argsort(
            list(testSampleFeatureSV[test_idx].values())).tolist()
        sortResult_aft = [sortResult_aft.index(k) \
                          for k in testSampleFeatureSV[test_idx].keys()]
        
        sortResult_change.append(
            np.mean(
                np.abs(np.array(sortResult_aft)-\
                       np.array(sortResult_bef))/\
                    len(testSampleFeatureSV[test_idx])
                    ))
        print('(aft PPM) sample %s\'s SV sortition:'%test_idx, 
              sortResult_change[-1],
              sortResult_bef, sortResult_aft)
        
    sortResult_change = np.mean(sortResult_change)
    print('sortResult_change: ',sortResult_change)
    
def FIA(SV_args):
    # Algorithm 1: The Attack with an Auxiliary Dataset
    # 60% training set, 20% test set (auxiliary set for training MLP), 
    # 20% validation set (query set producing MAE)
    SV_args.dataNormalize = True
    #SV_args.scannedIter_maxNum=50
    task = Task(SV_args) # FA task
    auxiliary_index = np.random.choice(range(len(task.Tst)),
                                       int(len(task.Tst)/2),
                                       replace = False).tolist()
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    
    #compute SV
    _, _, testSampleFeatureSV, testSampleFeatureSV_ref = FIA_logRead(SV_args)
    task.selected_test_samples = set(task.selected_test_samples)-\
                                 set(testSampleFeatureSV.keys())
    if len(task.selected_test_samples)>0:
        task.run()
    task.testSampleFeatureSV.update(testSampleFeatureSV)
    task.testSampleFeatureSV_var.update(testSampleFeatureSV_ref)
    if SV_args.privacy_protection_measure != None:
        FIA_addPrivacyProtection(SV_args, task.testSampleFeatureSV,
                                 task.testSampleFeatureSV_var)
        
    
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                    for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                     for test_idx in validation_index]
    
    auxiliary_data = ImageDataset(
        torch.FloatTensor(auxiliary_SV), 
        torch.FloatTensor(task.X_test[auxiliary_index]), 
        len(auxiliary_index), range(len(auxiliary_index)))
    validation_data = ImageDataset(
        torch.FloatTensor(validation_SV), 
        torch.FloatTensor(task.X_test[validation_index]), 
        len(validation_index), range(len(validation_index)))
    
    attackModel = LinearAttackModel(len(task.players))
    print('attackModel: ', attackModel)
    attackModel = DNNTrain(attackModel, auxiliary_data, 
                           epoch=100, batch_size=10, lr=0.005, 
                           loss_func=torch.nn.L1Loss(),
                           momentum=0, weight_decay=0, max_norm=0, 
                           validation_metric='tst_MAE', loss_print=True)
    
    MAE = DNNTest(attackModel, validation_data, test_bs=128, metric='tst_MAE',
                  pred_print=True) # mean absolute error
    print('FIA MAE: ', MAE)
    return MAE

def FIA_noAttackModelTrain(SV_args, random_mode='auxilliary'):
    # Algorithm 2: The Attack without an Auxiliary Dataset
    # M_c=30, tau=0.4, r=max S- min S, gagy=r/5
    #SV_args.scannedIter_maxNum=50
    SV_args.dataNormalize = True
    task = Task(SV_args) # FA task
    randomTestData, auxiliary_index,\
    testSampleFeatureSV, testSampleFeatureSV_ref = FIA_logRead(SV_args)
    if len(auxiliary_index)<=0:
        auxiliary_index = np.random.choice(range(len(task.Tst)),
                                           int(len(task.Tst)/2),
                                           replace = False).tolist()
    # generate random data
    if random_mode == 'uniform':
        print('model ', type(task.model), '---',
              'replace %s with random samples (uniform)...'%auxiliary_index)
        if 'Nets' in str(type(task.model)):
            task.Tst.dataset[auxiliary_index] = torch.FloatTensor(
                np.random.rand(
                    *task.Tst.dataset[auxiliary_index].shape))
            print('random data: ', task.Tst.dataset[auxiliary_index] )
        else:
            task.X_test[auxiliary_index] = np.random.rand(
                *task.X_test[auxiliary_index].shape)
            print('random data: ', task.X_test[auxiliary_index] )
            
    elif random_mode == 'normal':
        print('model ', type(task.model), '---',
              'replace %s with random samples (normal)...'%auxiliary_index)
        if 'Nets' in str(type(task.model)):
            task.Tst.dataset[auxiliary_index] = torch.FloatTensor(
                np.random.normal(
                    0.5,0.25, task.Tst.dataset[auxiliary_index].shape))
            print('random data: ', task.Tst.dataset[auxiliary_index] )
        else:
            task.X_test[auxiliary_index] = np.random.normal(
                0.5,0.25,task.X_test[auxiliary_index].shape)
            print('random data: ', task.X_test[auxiliary_index] )
    else:
        # use true data samples as the randomly-generated data sample
        pass
    for idx in testSampleFeatureSV.keys():
        if idx not in auxiliary_index:
            continue
        if 'torch' in str(type(task.model)):
            task.Tst.dataset[idx] = torch.FloatTensor(randomTestData[idx])
        else:
            task.X_test[auxiliary_index] = randomTestData[idx]
    #print(task.Tst.dataset)
    task.randomSet = auxiliary_index
    
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    print('validation_index:', validation_index, '\n',
          'auxiliary_index', auxiliary_index)
    
    # compute SV
    #task.run()
    task.selected_test_samples = set(task.selected_test_samples)-\
                                 set(testSampleFeatureSV.keys())
    if len(task.selected_test_samples)>0:
        task.run()
    task.testSampleFeatureSV.update(testSampleFeatureSV)
    task.testSampleFeatureSV_var.update(testSampleFeatureSV_ref)
    if SV_args.privacy_protection_measure != None:
        FIA_addPrivacyProtection(SV_args, task.testSampleFeatureSV,
                                 task.testSampleFeatureSV_var)
        
        
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                    for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                     for test_idx in validation_index]
        
    # start inference
    m_c=30
    r=np.array(auxiliary_SV).max()-np.array(auxiliary_SV).min()
    gagy = r/100
    tau=0.4
    print('number of reference samples:', m_c)
    print('threshold for feature SV difference:',r, gagy)
    print('threshold for feature max diff in references:',tau)
    predictions = np.zeros(task.X_test[validation_index].shape)
    num_unsuccess = 0
    for validation_data_idx in range(len(validation_SV)):
        for feature_idx in range(len(validation_SV[validation_data_idx])):
            
            diff_reference = dict()
            for reference_data_idx  in range(len(auxiliary_SV)):
                diff_reference[reference_data_idx]=np.abs(
                    validation_SV[validation_data_idx][feature_idx]-\
                    auxiliary_SV[reference_data_idx][feature_idx])
            diff_reference = dict(sorted(diff_reference.items(), 
                                         key=lambda item: item[1]))
            references = []
            for item  in diff_reference.items():
                if len(references)<m_c or item[1] < gagy:
                    references.append(
                        task.X_test[auxiliary_index[item[0]],feature_idx])
            if max(references)-min(references)>tau:
                #not to predict
                num_unsuccess += 1
                #print(validation_data_idx, feature_idx, ":", references)
            #else:
            predictions[validation_data_idx, feature_idx]=np.mean(references)
    MAE = F.l1_loss(torch.FloatTensor(predictions), 
                    torch.FloatTensor(task.X_test[validation_index]))
    print('FIA MAE: ', MAE)
    print('FIA SR: ', num_unsuccess, (validation_data_idx+1)*(feature_idx+1),
          1-num_unsuccess/((validation_data_idx+1)*(feature_idx+1)))
    return MAE



if __name__ == '__main__':    
    args = args_parser()
    if args.log_file!='':
        old_stdout = sys.stdout
        file = open(args.log_file, 'w')
        sys.stdout = file
    print('Experiment arguemtns: ', args)
    
    if args.task == 'DV':
        from data_valuation import Task 
    elif args.task == 'FA':
        from feature_attribution import Task 
    else:
        print('The task name may be wrong or ',
              'has not yet been considered in attack experiments!')
        sys.exit() 
        
    if args.attack_type == 'MIA':
        MIA(args.maxIter_in_MIA, args)
    elif args.attack_type == 'FIA':
        FIA(args)
    elif args.attack_type == 'FIA_withoutAuxilliary_uniform':
        FIA_noAttackModelTrain(args, random_mode='uniform')
    elif args.attack_type =='FIA_withoutAuxilliary_normal':
        FIA_noAttackModelTrain(args, random_mode='normal')
    elif args.attack_type =='FIA_noAttackModel':
        FIA_noAttackModelTrain(args, random_mode='auxilliary')
    else:
        print('The attack name may be wrong or ',
              'has not yet been considered in attack experiments!')
        sys.exit() 
    
    # Task terminated!
    sys.stdout.flush()
    if args.log_file!='':
        sys.stdout = old_stdout 
        file.close()
    sys.exit() 
