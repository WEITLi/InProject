# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import percentileofscore
import argparse

def concat_combination(data, window_size = 3, dname = 'cert'):
    '''
    目的: 将一个时间点的数据与其之前几个时间点的数据连接起来，形成一个更宽的特征向量。
    如何工作:
    对于每个用户，它会取当前时间点（比如某一天或某一会话）的特征。
    然后，它会回顾前 window_size - 1 个时间点的数据。
    将这些过去时间点的特征和当前时间点的特征水平堆叠（concatenate）起来。
    例如，如果 window_size = 3，它会把 t-2 时刻的特征、t-1 时刻的特征和 t 时刻的特征拼接在一起。
    信息列（如用户ID、时间戳、角色等）会被保留并附加到新的特征向量末尾。
    输出: 一个新的 DataFrame，其中每一行代表一个时间点，但特征维度是原始特征维度的 window_size 倍，外加原始的信息列。
    '''

    if dname == 'cert':
        info_cols = ['sessionid','day','week',"starttime", "endtime",
                     'user', 'project', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 
                     'O', 'C', 'E', 'A', 'N', 'insider']
        
    combining_features = [ f for f in data.columns if f not in info_cols]
    info_features = [f for f in data.columns if f in info_cols]
    
    data_info = data[info_features].values
    
    data_combining_features = data[combining_features].values
    useridx = data['user'].values
    
    userset = set(data['user'])

    cols = []
    for shiftrange in range(window_size-1,0,-1):
        cols += [str(-shiftrange) + '_' + f for f in combining_features]
    cols += combining_features + info_features
    
    combined_data = []
    for u in userset:
        data_cf_u = data_combining_features[useridx == u, ]
        
        data_cf_u_shifted = []
        for shiftrange in range(window_size-1,0,-1):
            data_cf_u_shifted.append(np.roll(data_cf_u, shiftrange, axis = 0))
        
        data_cf_u_shifted.append(data_cf_u)
        data_cf_u_shifted.append(data_info[useridx==u, ])
        
        combined_data.append(np.hstack(data_cf_u_shifted)[window_size:,])
    
    combined_data = pd.DataFrame(np.vstack(combined_data), columns=cols)
    
    return combined_data


def subtract_combination_uworker(u, alluserdict, dtype, calc_type, window_size, udayidx, udata, uinfo, uorg):
    '''
    这个函数的主要目的是计算当前时间点的特征与前一个时间窗口内特征的差异或相对位置。它支持并行处理，以加速计算。具体步骤如下：
    参数说明:
    u: 用户标识符。
    alluserdict: 用于存储每个用户的计算结果的字典。
    dtype: 数据类型（'day' 或 'week'）。
    calc_type: 计算类型（'meandiff', 'meddiff', 'percentile'）。
    window_size: 时间窗口大小。
    udayidx, udata, uinfo, uorg: 用户的时间索引、特征数据、信息数据和原始特征数据。
    计算逻辑:
    对于每个用户和每个时间点 t，查看t之前的 window_size 个时间点。
    根据 calc_type 参数，执行不同的计算：
    'meandiff': 计算当前时间点的特征值与前一个窗口内特征值的均值之差。
    'meddiff': 计算当前时间点的特征值与前一个窗口内特征值的中位数之差。
    'percentile': 计算当前时间点的每个特征值在前一个窗口内所有对应特征值中的百分位排名。
    输出:
    将计算结果存储在 alluserdict 中，每个用户的结果是一个二维数组，表示特征的动态变化。
    '''
    
    if u%200==0: 
        print(u)

    data_out = []
     
    if dtype in ['day', 'week']:
        
        for i in range(len(udayidx)):
            t = udayidx[i]
            if dtype in ['day','week']: min_idx = min(udayidx)+window_size
            
            if t>=min_idx:
                if calc_type == 'meandiff':
                    prevdata = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if len(prevdata) < 1: continue 
                    window_mean = np.mean(prevdata, axis = 0)
                    data_out.append(np.concatenate((udata[i] - window_mean, uorg[i,:], uinfo[i,:])))
                   
                if calc_type == 'meddiff':
                    prevdata = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if len(prevdata) < 1: continue 
                    window_med = np.median(prevdata, axis = 0)
                    data_out.append(np.concatenate((udata[i] - window_med, uorg[i,:], uinfo[i,:])))
                elif calc_type == 'percentile':
                    window = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if window.shape[0] < 1: continue
                    percentile_i = [percentileofscore(window[:,j], udata[i,j], 'mean') - 50 for j in range(window.shape[1])]
                    data_out.append(np.concatenate((percentile_i , uorg[i,:], uinfo[i,:])))
                    
    if len(data_out) > 0: alluserdict[u] = np.vstack(data_out)

def subtract_percentile_combination(data, dtype, calc_type = 'percentile', window_size = 7, dname = 'cert', parallel = True):
    '''
    Combine data to generate different temporal representations
    window_size: window size by days (for CERT data)
    参数说明:
    data: 输入数据。
    dtype: 数据类型（'day' 或 'week'）。
    calc_type: 计算类型（默认为 'percentile'）。
    window_size: 时间窗口大小。
    dname: 数据集名称（默认为 'cert'）。
    parallel: 是否并行处理。
    处理逻辑:
    根据 dname 和 dtype，确定信息列和需要保留的原始特征列。
    将数据按用户分组，并根据 dtype 确定时间索引。
    如果 parallel 为真，使用多进程并行处理每个用户的数据，调用 subtract_combination_uworker 函数。
    如果 parallel 为假，逐个处理每个用户的数据。
    输出:
    返回一个新的 DataFrame，其中特征列代表了当前数据点相对于其历史窗口的动态变化。
    '''
    if dname == 'cert':
        info_cols = ['sessionid','day','week',"starttime", "endtime", 
                     'user', 'project', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 
                     'O', 'C', 'E', 'A', 'N', 'insider','subs_ind']
        keep_org_cols = ["pc", "isworkhour", "isafterhour", "isweekday", "isweekend", "isweekendafterhour", "n_days", 
                         "duration", "n_concurrent_sessions", "start_with", "end_with", "ses_start", "ses_end"]
        
    combining_features = [ f for f in data.columns if f not in info_cols]
    info_features = [f for f in data.columns if f in info_cols] 
    keep_org_features = [f for f in data.columns if f in keep_org_cols]
    
    data_info = data[info_features].values
    data_org = data[keep_org_features].values
    data_combining_features = data[combining_features].values
    useridx = data['user'].values
    if dtype in ['day']: dayidx = data['day'].values
    if dname == 'cert': weekidx = data['week'].values
    
    userset = set(data['user'])
    
    if dtype == 'week': 
        window_size = np.floor(window_size/7)
        idx = weekidx
    elif dtype in ['day']: idx = dayidx

    if parallel:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for u in userset:
            udayidx = idx[useridx==u]
            udata = data_combining_features[useridx==u, ]
            uinfo = data_info[useridx==u, ]
            uorg = data_org[useridx==u, ]
            p = multiprocessing.Process(target=subtract_combination_uworker, args=(u, return_dict, dtype, calc_type,
                                                                                    window_size, udayidx,
                                                                                    udata, uinfo, uorg))
            jobs.append(p)
            p.start()
    
        for proc in jobs:
            proc.join()
    else:
        return_dict = {}
        for u in userset:
            udayidx = idx[useridx==u]
            udata = data_combining_features[useridx==u, ]
            uinfo = data_info[useridx==u, ]
            uorg = data_org[useridx==u, ]
            subtract_combination_uworker(u, return_dict, dtype, calc_type,
                                        window_size, udayidx,
                                        udata, uinfo, uorg)

    combined_data = pd.DataFrame(np.vstack([return_dict[ri] for ri in return_dict.keys()]), columns=combining_features+['org_'+f for f in keep_org_features] + info_features)
    
    return combined_data


if __name__ == "__main__":    
    # 参数解析: 使用 argparse 来接收命令行参数，允许用户指定：
    # --representation: 要提取的时间表示类型（'concat', 'percentile', 'meandiff', 'meddiff', 或 'all' 来提取所有类型）。默认为 'percentile'。
    # --file_input: 输入的 CSV 文件名（例如 week-r5.2.csv.gz）。
    # --window_size: 用于差值/百分位计算的时间窗口大小（以天为单位）。默认为30。
    # --num_concat: 用于连接组合的连续数据点数量。默认为3。

    parser=argparse.ArgumentParser()
    parser.add_argument('--representation', help='Data representation to extract (concat, percentile, meandiff, mediandiff). Default: percentile', 
                        type= str, default = 'percentile')
    # parser.add_argument('--file_input', help='CERT input file name. Default: week-r5.2.csv.gz', type= str, default= 'week-r5.2.csv.gz')  
    parser.add_argument('--file_input', help='CERT input file name. Default: week-r5.2.csv.gz', type= str, default= 'week-r4.2.csv.gz')  
    parser.add_argument('--window_size', help='Window size for percentile or mean/median difference representation. Default: 30', 
                        type = int, default=30)
    parser.add_argument('--num_concat', help='Number of data points for concatenation. Default: 3', 
                        type = int, default=3)
    args=parser.parse_args()    
    
    print('If "too many opened files", or "ForkAwareLocal" error, run ulimit command, e.g. "$ulimit -n 10000" to increase the limit first')
    if args.representation == 'all':
        reps = ['concat', 'percentile','meandiff','meddiff']
    elif args.representation in ['concat', 'percentile','meandiff','meddiff']:
        reps = [args.representation]
        
    fileName = (args.file_input).replace('.csv','').replace('.gz','')
    if 'day' in fileName:
        data_type = 'day'
    elif 'week' in fileName:
        data_type = 'week'

    # 数据加载: 读取指定的输入 CSV 文件。
    s = pd.read_csv(f'{args.file_input}')
    
    # 特征提取与保存:
    # 根据用户选择的表示类型 (rep)：
    # 如果选择的是 'percentile', 'meandiff', 或 'meddiff'，则调用 subtract_percentile_combination 函数。
    # 如果选择的是 'concat'，则调用 concat_combination 函数。
    # 将生成的新特征 DataFrame 保存为 pickle 文件（.pkl），文件名会包含原始文件名、表示类型和窗口大小等信息（例如 week-r5.2-percentile30.pkl）。

    for rep in reps:
        if rep in ['percentile','meandiff','meddiff']:
            s1 = subtract_percentile_combination(s, data_type, calc_type = rep, window_size = args.window_size, dname='cert')
            s1.to_pickle(f'{fileName}-{rep}{args.window_size}.pkl')
        else:
            s1 = concat_combination(s, window_size = args.num_concat, dname = 'cert')
            s1.to_pickle(f'{fileName}-{rep}{args.num_concat}.pkl')
    
