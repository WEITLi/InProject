#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lcd
"""
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import subprocess
from joblib import Parallel, delayed

def time_convert(inp, mode, real_sd = '2010-01-02', sd_monday= "2009-12-28"):
    """
    时间格式转换函数
    参数:
    - inp: 输入时间
    - mode: 转换模式,包括:
        e2t: epoch时间戳转字符串时间
        t2e: 字符串时间转epoch时间戳
        t2dt: 字符串时间转datetime对象
        等多种转换模式
    - real_sd: 实际开始日期
    - sd_monday: 开始周一日期
    """
    if mode == 'e2t':
        return datetime.fromtimestamp(inp).strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 't2e':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').strftime('%s')
    elif mode == 't2dt':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S')
    elif mode == 't2date':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').strftime("%Y-%m-%d")
    elif mode == 'dt2t':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2W':
        return int(inp.strftime('%W'))
    elif mode == 'dt2d':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2date':
        return inp.strftime("%Y-%m-%d")
    elif mode =='dt2dn': #datetime to day number
        startdate = datetime.strptime(sd_monday,'%Y-%m-%d')
        return (inp - startdate).days
    elif mode =='dn2epoch': #datenum to epoch
        dt = datetime.strptime(sd_monday,'%Y-%m-%d') + timedelta(days=inp)
        return int(dt.timestamp())
    elif mode =='dt2wn': #datetime to week number
        startdate = datetime.strptime(real_sd,'%Y-%m-%d')
        return (inp - startdate).days//7
    elif mode =='t2wn': #datetime to week number
        startdate = datetime.strptime(real_sd,'%Y-%m-%d')
        return (datetime.strptime(inp, '%m/%d/%Y %H:%M:%S') - startdate).days//7
    elif mode == 'dt2wd':
        return int(inp.strftime("%w"))
    elif mode == 'm2dt':
        return datetime.strptime(inp, "%Y-%m")
    elif mode == 'datetoweekday':
        return int(datetime.strptime(inp,"%Y-%m-%d").strftime('%w'))
    elif mode == 'datetoweeknum':
        w0 = datetime.strptime(sd_monday,"%Y-%m-%d")
        return int((datetime.strptime(inp,"%Y-%m-%d") - w0).days / 7)
    elif mode == 'weeknumtodate':
        startday = datetime.strptime(sd_monday,"%Y-%m-%d")
        return startday+timedelta(weeks = inp)
    
def add_action_thisweek(act, columns, lines, act_handles, week_index, stop, firstdate, dname = 'r5.2'):
    """
    将一周内的特定类型活动添加到数据框中
    参数:
    - act: 活动类型(email,file,http等)
    - columns: 列名
    - lines: 当前读取的行
    - act_handles: 文件句柄
    - week_index: 周索引
    - stop: 停止标志
    - firstdate: 开始日期
    - dname: 数据集名称
    返回: 包含该周活动的DataFrame
    """
    thisweek_act = []
    while True:
        if not lines[act]: 
            stop[act] = 1
            break
        if dname in ['r6.1','r6.2'] and act in ['email', 'file','http'] and '"' in lines[act]:
            tmp = lines[act]
            firstpart = tmp[:tmp.find('"')-1]
            content = tmp[tmp.find('"')+1:-1]
            tmp = firstpart.split(',') + [content]
        else:
            tmp = lines[act].split(',')
        if time_convert(tmp[1], 't2wn', real_sd= firstdate) == week_index:
            thisweek_act.append(tmp)
        else:
            break
        lines[act] = act_handles[act].readline()
    df = pd.DataFrame(thisweek_act, columns=columns)
    df['type']= act
    df.index = df['id']
    df.drop('id', axis = 1, inplace = True)
    return df

def combine_by_timerange_pandas(dname = 'r4.2'):
    """
    按周合并所有类型的活动数据
    参数:
    - dname: 数据集名称
    功能:
    - 读取device,email,file,http,logon等活动数据
    - 按周组织数据并保存到pickle文件
    """

    allacts =  ['device','email','file', 'http','logon']
    firstline = str(subprocess.check_output(['head', '-2', 'http.csv'])).split('\\n')[1]
    firstdate = time_convert(firstline.split(',')[1],'t2dt')
    firstdate = firstdate - timedelta(int(firstdate.strftime("%w")))
    firstdate = time_convert(firstdate, 'dt2date')
    week_index = 0
    act_handles = {}
    lines = {}
    stop = {}
    for act in allacts:
        act_handles[act] = open(act+'.csv','r')
        next(act_handles[act],None) #skip header row
        lines[act] = act_handles[act].readline()
        stop[act] = 0 # store stop value indicating if all of the file has been read
    while sum(stop.values()) < 5:
        thisweekdf = pd.DataFrame()
        for act in allacts:
            if 'email' == act:
                if dname in ['r4.1','r4.2']:
                    columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', '#att', 'content']
                if dname in ['r6.1','r6.2','r5.2','r5.1']:
                    columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity', 'size', 'att', 'content']     
            elif 'logon' == act:
                columns = ['id', 'date', 'user', 'pc', 'activity']
            elif 'device' == act:
                if dname in ['r4.1','r4.2']:
                    columns = ['id', 'date', 'user', 'pc', 'activity']
                if dname in ['r5.1','r5.2','r6.2','r6.1']:
                    columns = ['id', 'date', 'user', 'pc', 'content', 'activity']
            elif 'http' == act:
                if dname in ['r6.1','r6.2']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'activity', 'content']
                if dname in ['r5.1','r5.2','r4.2','r4.1']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'content']
            elif 'file' == act:
                if dname in ['r4.1','r4.2']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'content']
                if dname in ['r5.2','r5.1','r6.2','r6.1']: columns = ['id', 'date', 'user', 'pc', 'url/fname','activity','to','from','content']
            
            df = add_action_thisweek(act, columns, lines, act_handles, week_index, stop, firstdate, dname=dname)
            # thisweekdf = thisweekdf.append(df, sort=False)
            thisweekdf = pd.concat([thisweekdf, df], sort=False, ignore_index=True) # 使用 pd.concat 替代 append
        
        thisweekdf['date'] = thisweekdf['date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
        thisweekdf.to_pickle("DataByWeek/"+str(week_index)+".pickle")
        week_index += 1

##############################################################################

def process_user_pc(upd, roles): #figure out  which PC belongs to which user
    """
    处理用户-PC对应关系
    参数:
    - upd: 用户-PC数据
    - roles: 用户角色信息
    功能:
    - 确定每个用户的主要PC
    - 识别共享PC
    - 处理IT管理员特殊情况
    """

    upd['sharedpc'] = None
    upd['npc'] = upd['pcs'].apply(lambda x: len(x))
    upd.at[upd['npc']==1,'pc'] = upd[upd['npc']==1]['pcs'].apply(lambda x: x[0])
    multiuser_pcs = np.concatenate(upd[upd['npc']>1]['pcs'].values).tolist()
    set_multiuser_pc = list(set(multiuser_pcs))
    count = {}
    for pc in set_multiuser_pc:
        count[pc] = multiuser_pcs.count(pc)
    for u in upd[upd['npc']>1].index:
        sharedpc = upd.loc[u]['pcs']
        count_u_pc = [count[pc] for pc in upd.loc[u]['pcs']]
        the_pc = count_u_pc.index(min(count_u_pc))
        upd.at[u,'pc'] = sharedpc[the_pc]
        if roles.loc[u] != 'ITAdmin':
            sharedpc.remove(sharedpc[the_pc])
            upd.at[u,'sharedpc']= sharedpc
    return upd

def getuserlist(dname = 'r4.2', psycho = True):
    """
    获取用户列表及其属性
    参数:
    - dname: 数据集名称
    - psycho: 是否包含心理测量数据
    功能:
    - 读取LDAP用户信息
    - 处理用户离职信息
    - 合并心理测量数据(如果有)
    - 确定用户PC信息
    """
    allfiles =  ['LDAP/'+f1 for f1 in os.listdir('LDAP') if os.path.isfile('LDAP/'+f1)]
    alluser = {}
    alreadyFired = []
    
    for file in allfiles:
        af = (pd.read_csv(file,delimiter=',')).values
        employeesThisMonth = []    
        for i in range(len(af)):
            employeesThisMonth.append(af[i][1])
            if af[i][1] not in alluser:
                alluser[af[i][1]] = af[i][0:1].tolist() + af[i][2:].tolist() + [file.split('.')[0] , np.nan]

        firedEmployees = list(set(alluser.keys()) - set(alreadyFired) - set(employeesThisMonth))
        alreadyFired = alreadyFired + firedEmployees
        for e in firedEmployees:
            alluser[e][-1] = file.split('.')[0]
    
    if psycho and os.path.isfile("psychometric.csv"):

        p_score = pd.read_csv("psychometric.csv",delimiter = ',').values
        for id in range(len(p_score)):
            alluser[p_score[id,1]] = alluser[p_score[id,1]]+ list(p_score[id,2:])
        df  = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in ['r4.1','r4.2']:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
        elif dname in ['r5.2','r5.1','r6.2','r6.1']:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
    else:
        df  = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in ['r4.1','r4.2']:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']
        elif dname in ['r5.2','r5.1','r6.2','r6.1']:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']

    df['pc'] = None
    for i in df.index:
        if type(df.loc[i]['sup']) == str:
            sup = df[df['uname'] == df.loc[i]['sup']].index[0]
        else:
            sup = None
        df.at[i,'sup'] = sup
        
    #read first 2 weeks to determine each user's PC
    w1 = pd.read_pickle("DataByWeek/1.pickle")
    w2 = pd.read_pickle("DataByWeek/2.pickle")
    user_pc_dict = pd.DataFrame(index=df.index)
    user_pc_dict['pcs'] = None  
  
    for u in df.index:
        pc = list(set(w1[w1['user']==u]['pc']) & set(w2[w2['user']==u]['pc']))
        user_pc_dict.at[u,'pcs'] = pc
    upd = process_user_pc(user_pc_dict, df['role'])
    df['pc'] = upd['pc']
    df['sharedpc'] = upd['sharedpc']
    return df

        
def get_mal_userdata(data = 'r4.2', usersdf = None):
    """
    获取恶意用户数据
    参数:
    - data: 数据集名称
    - usersdf: 用户数据框
    功能:
    - 读取内部威胁者信息
    - 标记恶意活动时间段
    - 记录恶意活动详情
    """
    # 手动实现
    # if not os.path.isdir('answers'):
    #     os.system('wget https://kilthub.cmu.edu/ndownloader/files/24857828 -O answers.tar.bz2')
    #     os.system('tar -xjvf answers.tar.bz2')
    
    listmaluser = pd.read_csv("answers/insiders.csv")
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset']==data.replace("r","")]
    #for r6.2, new time in scenario 4 answer is incomplete.
    if data == 'r6.2': listmaluser.at[listmaluser['scenario']==4,'start'] = '02'+listmaluser[listmaluser['scenario']==4]['start']
    listmaluser[['start','end']] = listmaluser[['start','end']].applymap(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
    
    if type(usersdf) != pd.core.frame.DataFrame:
        usersdf = getuserlist(data)
    usersdf['malscene']=0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['malacts'] = None
    
    for i in listmaluser.index:
        usersdf.loc[listmaluser['user'][i], 'mstart'] = listmaluser['start'][i]
        usersdf.loc[listmaluser['user'][i], 'mend'] = listmaluser['end'][i]
        usersdf.loc[listmaluser['user'][i], 'malscene'] = listmaluser['scenario'][i]
        
        if data in ['r4.2', 'r5.2']:
            malacts = open(f"answers/r{listmaluser['dataset'][i]}-{listmaluser['scenario'][i]}/"+
                       listmaluser['details'][i],'r').read().strip().split("\n")
        else: #only 1 malicious user, no folder
            malacts = open("answers/"+ listmaluser['details'][i],'r').read().strip().split("\n")
        
        malacts = [x.split(',') for x in malacts]

        mal_users = np.array([x[3].strip('"') for x in malacts])
        mal_act_ids =  np.array([x[1].strip('"') for x in malacts])
        
        usersdf.at[listmaluser['user'][i], 'malacts'] = mal_act_ids[mal_users==listmaluser['user'][i]]
                    
    return usersdf

##############################################################################

def is_after_whour(dt): #Workhours assumed 7:30-17:30
    """判断是否在工作时间之后"""
    wday_start = datetime.strptime("7:30", "%H:%M").time()
    wday_end = datetime.strptime("17:30", "%H:%M").time()
    dt = dt.time()
    if dt < wday_start or dt > wday_end:
        return True
    return False
      
def is_weekend(dt):
    """判断是否是周末"""
    if dt.strftime("%w") in ['0', '6']:
        return True
    return False   
    
def email_process(act, data = 'r4.2', separate_send_receive = True):
    """处理邮件活动特征"""
    receivers = act['to'].split(';')
    if type(act['cc']) == str:
        receivers = receivers + act['cc'].split(";")
    if type(act['bcc']) == str:
        bccreceivers = act['bcc'].split(";")   
    else:
        bccreceivers = []
    exemail = False
    n_exdes = 0
    for i in receivers + bccreceivers:
        if 'dtaa.com' not in i:
            exemail = True
            n_exdes += 1

    n_des = len(receivers) + len(bccreceivers)
    Xemail = 1 if exemail else 0
    n_bccdes = len(bccreceivers)
    exbccmail = 0
    email_text_len = len(act['content'])
    email_text_nwords = act['content'].count(' ') + 1
    for i in bccreceivers:
        if 'dtaa.com' not in i:
            exbccmail = 1
            break

    if data in ['r5.1','r5.2','r6.1','r6.2']:
        send_mail = 1 if act['activity'] == 'Send' else 0
        receive_mail = 1 if act['activity'] in ['Receive','View'] else 0
        
        atts = act['att'].split(';')
        n_atts = len(atts)
        size_atts = 0
        att_types = [0,0,0,0,0,0]
        att_sizes = [0,0,0,0,0,0]
        for att in atts:
            if '.' in att:
                tmp = file_process(att, filetype='att')
                att_types = [sum(x) for x in zip(att_types,tmp[0])]
                att_sizes = [sum(x) for x in zip(att_sizes,tmp[1])]
                size_atts +=sum(tmp[1])
        return [send_mail, receive_mail, n_des, n_atts, Xemail, n_exdes, 
                n_bccdes, exbccmail, int(act['size']), email_text_len, 
                email_text_nwords] + att_types + att_sizes
    elif data in ['r4.1','r4.2']:
        return [n_des, int(act['#att']), Xemail, n_exdes, n_bccdes, exbccmail, 
                int(act['size']), email_text_len, email_text_nwords]
        
def http_process(act, data = 'r4.2'): 
    """处理HTTP活动特征"""
    # basic features:
    url_len = len(act['url/fname'])
    url_depth = act['url/fname'].count('/')-2
    content_len = len(act['content'])
    content_nwords = act['content'].count(' ')+1
    
    domainname = re.findall("//(.*?)/", act['url/fname'])[0]
    domainname.replace("www.","")
    dn = domainname.split(".")
    if len(dn) > 2 and not any([x in domainname for x in ["google.com", '.co.uk', '.co.nz', 'live.com']]):
        domainname = ".".join(dn[-2:])

    # other 1, socnet 2, cloud 3, job 4, leak 5, hack 6
    if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
        r = 3
    elif domainname in ['wikileaks.org','freedom.press','theintercept.com']:
        r = 5
    elif domainname in ['facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com',
                        'flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com',
                        'tumblr.com','quora.com','vine.co','match.com','t.co']:
        r = 2
    elif domainname in ['indeed.com','monster.com', 'careerbuilder.com','simplyhired.com']:
        r = 4
    
    elif ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) \
    or ('aol.com' in domainname and ("recruit" in act['url/fname'] or "job" in act['url/fname'])):
        r = 4
    elif (domainname in ['webwatchernow.com','actionalert.com', 'relytec.com','refog.com','wellresearchedreviews.com',
                         'softactivity.com', 'spectorsoft.com','best-spy-soft.com']):
        r = 6
    elif ('keylog' in domainname):
        r = 6
    else:
        r = 1
    if data in ['r6.1','r6.2']:
        http_act_dict = {'www visit': 1, 'www download': 2, 'www upload': 3}
        http_act = http_act_dict.get(act['activity'].lower(), 0)
        return [r, url_len, url_depth, content_len, content_nwords, http_act]
    else:
        return [r, url_len, url_depth, content_len, content_nwords]
        
def file_process(act, complete_ul = None, data = 'r4.2', filetype = 'act'):
    """处理文件活动特征"""
    if filetype == 'act':
        ftype = act['url/fname'].split(".")[1]
        disk = 1 if act['url/fname'][0] == 'C' else 0
        if act['url/fname'][0] == 'R': disk = 2
        file_depth = act['url/fname'].count('\\')
    elif filetype == 'att': #attachments
        tmp = act.split('.')[1]
        ftype = tmp[:tmp.find('(')]
        attsize = int(tmp[tmp.find("(")+1:tmp.find(")")])
        r = [[0,0,0,0,0,0], [0,0,0,0,0,0]]
        if ftype in ['zip','rar','7z']:
            ind = 1
        elif ftype in ['jpg', 'png', 'bmp']:
            ind = 2
        elif ftype in ['doc','docx', 'pdf']:
            ind = 3
        elif ftype in ['txt','cfg', 'rtf']:
            ind = 4
        elif ftype in ['exe', 'sh']:
            ind = 5
        else:
            ind = 0
        r[0][ind] = 1
        r[1][ind] = attsize
        return r

    fsize = len(act['content'])
    f_nwords = act['content'].count(' ')+1
    if ftype in ['zip','rar','7z']:
        r = 2
    elif ftype in ['jpg', 'png', 'bmp']:
        r = 3
    elif ftype in ['doc','docx', 'pdf']:
        r = 4
    elif ftype in ['txt','cfg','rtf']:
        r = 5
    elif ftype in ['exe', 'sh']:
        r = 6
    else:
        r = 1
    if data in ['r5.2','r5.1', 'r6.2','r6.1']:
        to_usb = 1 if act['to'] == 'True' else 0
        from_usb = 1 if act['from'] == 'True' else 0
        file_depth = act['url/fname'].count('\\')
        file_act_dict = {'file open': 1, 'file copy': 2, 'file write': 3, 'file delete': 4}
        if act['activity'].lower() not in file_act_dict: print(act['activity'].lower())
        file_act = file_act_dict.get(act['activity'].lower(), 0)
        return [r, fsize, f_nwords, disk, file_depth, file_act, to_usb, from_usb]
    elif data in ['r4.1','r4.2']:
        return [r, fsize, f_nwords, disk, file_depth]

def from_pc(act, ul):
    """
    判断用户活动发生在什么类型的电脑上
    参数:
    - act: 用户活动数据
    - ul: 用户列表数据
    返回:
    - 电脑类型代码(0-3)和电脑ID
    - 0: 用户自己的电脑
    - 1: 共享电脑
    - 2: 他人的电脑
    - 3: 主管的电脑
    """
    user_pc = ul.loc[act['user']]['pc']
    act_pc = act['pc']
    if act_pc == user_pc:
        return (0, act_pc) #using normal PC
    elif ul.loc[act['user']]['sharedpc'] is not None and act_pc in ul.loc[act['user']]['sharedpc']:
        return (1, act_pc)
    elif ul.loc[act['user']]['sup'] is not None and act_pc == ul.loc[ul.loc[act['user']]['sup']]['pc']:
        return (3, act_pc)
    else:
        return (2, act_pc)
    
def process_week_num(week, users, userlist = 'all', data = 'r4.2'):
    """
    处理一周内的用户活动数据,转换为数值特征
    
    这个函数是特征提取的核心部分，它将原始的用户活动数据（登录、文件操作、邮件、HTTP访问等）
    转换为机器学习可以使用的数值特征向量。
    
    参数:
    - week: 周数索引，用于读取对应周的数据文件
    - users: 用户信息数据框，包含用户属性和恶意用户标记
    - userlist: 要处理的用户列表，默认'all'表示处理所有用户
    - data: 数据集名称(r4.2, r5.2等)，不同数据集的特征维度不同
    
    功能:
    - 读取一周的活动数据（从DataByWeek文件夹）
    - 将每种活动类型转换为对应的数值特征
    - 标记恶意活动和内部威胁用户
    - 保存处理后的数值数据到NumDataByWeek文件夹
    
    输出:
    - 生成包含数值特征的pickle文件，用于后续的统计特征计算
    """
    
    # ========== 1. 初始化和数据准备 ==========
    # 创建用户ID到索引的映射字典，用于后续快速查找
    user_dict = {idx: i for (i, idx) in enumerate(users.index)}
    
    # 读取指定周的活动数据（之前由combine_by_timerange_pandas生成）
    acts_week = pd.read_pickle("DataByWeek/"+str(week)+".pickle")
    
    # 获取该周的时间范围，用于判断恶意活动时间窗口
    start_week, end_week = min(acts_week.date), max(acts_week.date)
    
    # 按时间排序，确保活动按时间顺序处理（对于connect/disconnect配对很重要）
    acts_week.sort_values('date', ascending = True, inplace = True)
    
    # ========== 2. 确定特征维度 ==========
    # 根据不同数据集版本设置特征向量的维度
    # r5.x版本有45列，r6.x版本有46列（多了HTTP活动类型），r4.x版本只有27列
    n_cols = 45 if data in ['r5.2','r5.1'] else 46
    if data in ['r4.2','r4.1']: n_cols = 27
    
    # 初始化存储所有用户该周活动特征的矩阵
    u_week = np.zeros((len(acts_week), n_cols))
    
    # 存储每个活动的元信息：[活动ID, PC ID, 时间戳]
    pc_time = []
    
    # 如果没有指定用户列表，则处理该周所有活跃用户
    if userlist == 'all':
        userlist = set(acts_week.user)
    
    # ========== 3. 按用户循环处理活动数据 ==========
    current_ind = 0  # 当前在总矩阵中的行索引
    
    for u in userlist:
        # 提取当前用户在该周的所有活动
        df_acts_u = acts_week[acts_week.user == u]
        
        # ========== 3.1 判断用户是否为恶意用户 ==========
        mal_u = 0  # 初始化为正常用户
        # 如果用户在恶意用户列表中（malscene > 0表示参与了某个恶意场景）
        if users.loc[u].malscene > 0:
            # 检查当前周是否在该用户的恶意活动时间窗口内
            if start_week <= users.loc[u].mend and users.loc[u].mstart <= end_week:
                mal_u = users.loc[u].malscene  # 记录恶意场景编号
        
        # ========== 3.2 活动类型标准化和映射 ==========
        # 获取所有活动类型（来自type列）
        list_uacts = df_acts_u.type.tolist()
        
        # 获取活动的具体操作（来自activity列，主要用于设备连接操作）
        list_activity = df_acts_u.activity.tolist()
        
        # 对于设备相关活动，将activity列的值转换为小写并标准化
        # 这里处理Logon/Logoff/Connect/Disconnect等设备操作
        list_uacts = [list_activity[i].strip().lower() if (type(list_activity[i])==str and list_activity[i].strip() in ['Logon', 'Logoff', 'Connect', 'Disconnect']) \
                        else list_uacts[i] for i in range(len(list_uacts))]
        
        # 将活动类型映射为数值编码
        # 1:登录, 2:登出, 3:设备连接, 4:设备断开, 5:HTTP访问, 6:邮件, 7:文件操作
        uacts_mapping = {'logon':1, 'logoff':2, 'connect':3, 'disconnect':4, 'http':5,'email':6,'file':7}
        list_uacts_num = [uacts_mapping[x] for x in list_uacts]

        # 初始化当前用户的特征矩阵
        oneu_week = np.zeros((len(df_acts_u), n_cols))
        oneu_pc_time = []  # 存储当前用户的活动元信息
        
        # ========== 3.3 逐个活动处理 ==========
        for i in range(len(df_acts_u)):
            # ========== 3.3.1 确定活动发生的PC类型 ==========
            # 调用from_pc函数判断活动发生在什么类型的电脑上
            # 返回值：0=自己的PC, 1=共享PC, 2=他人的PC, 3=主管的PC
            pc, _ = from_pc(df_acts_u.iloc[i], users)
            
            # ========== 3.3.2 确定活动发生的时间类型 ==========
            # 根据是否为周末和是否为工作时间，将时间分为4类
            if is_weekend(df_acts_u.iloc[i]['date']):
                if is_after_whour(df_acts_u.iloc[i]['date']):
                    act_time = 4  # 周末非工作时间
                else:
                    act_time = 3  # 周末工作时间
            elif is_after_whour(df_acts_u.iloc[i]['date']):
                act_time = 2  # 工作日非工作时间
            else:
                act_time = 1  # 工作日工作时间
            
            # ========== 3.3.3 初始化各类活动的特征向量 ==========
            # 根据数据集版本初始化不同维度的特征向量
            if data in ['r4.2','r4.1']:
                device_f = [0]              # 设备特征：USB连接时长
                file_f = [0, 0, 0, 0, 0]    # 文件特征：类型、大小、词数、磁盘、深度
                http_f = [0,0,0,0,0]        # HTTP特征：类型、URL长度、深度、内容长度、词数
                email_f = [0]*9             # 邮件特征：9维
            elif data in ['r5.2','r5.1','r6.2','r6.1']:
                device_f = [0,0]            # 设备特征：USB连接时长、文件树长度
                file_f = [0]*8              # 文件特征：8维（增加了USB传输标记）
                http_f = [0,0,0,0,0]        # HTTP特征：5维基础特征
                if data in ['r6.2','r6.1']:
                    http_f = [0,0,0,0,0,0]  # r6版本增加了HTTP活动类型
                email_f = [0]*23            # 邮件特征：23维（增加了附件相关特征）
            
            # ========== 3.3.4 根据活动类型提取具体特征 ==========
            if list_uacts[i] == 'file':
                # 文件操作：提取文件类型、大小、深度等特征
                file_f = file_process(df_acts_u.iloc[i], data = data)
                
            elif list_uacts[i] == 'email':
                # 邮件活动：提取收件人数量、附件、外部邮件等特征
                email_f = email_process(df_acts_u.iloc[i], data = data)
                
            elif list_uacts[i] == 'http':
                # HTTP访问：提取URL类型、长度、内容等特征
                http_f = http_process(df_acts_u.iloc[i], data=data)
                
            elif list_uacts[i] == 'connect':
                # ========== 设备连接：需要特殊处理，计算连接持续时间 ==========
                # 从当前活动开始，向后查找对应的断开连接活动
                tmp = df_acts_u.iloc[i:]
                
                # 查找同一用户在同一PC上的断开连接活动
                disconnect_acts = tmp[(tmp['activity'] == 'Disconnect\n') & \
                 (tmp['user'] == df_acts_u.iloc[i]['user']) & \
                 (tmp['pc'] == df_acts_u.iloc[i]['pc'])]
                
                # 查找同一用户在同一PC上的后续连接活动（用于处理异常情况）
                connect_acts = tmp[(tmp['activity'] == 'Connect\n') & \
                 (tmp['user'] == df_acts_u.iloc[i]['user']) & \
                 (tmp['pc'] == df_acts_u.iloc[i]['pc'])]
                
                # 计算连接持续时间
                if len(disconnect_acts) > 0:
                    distime = disconnect_acts.iloc[0]['date']
                    # 如果在断开之前又有连接，说明数据异常
                    if len(connect_acts) > 0 and connect_acts.iloc[0]['date'] < distime:
                        connect_dur = -1  # 标记为异常
                    else:
                        # 计算正常的连接持续时间（以秒为单位）
                        tmp_td = distime - df_acts_u.iloc[i]['date']
                        connect_dur = tmp_td.days*24*3600 + tmp_td.seconds
                else:
                    connect_dur = -1  # 没有找到断开连接活动，标记为异常
                    
                # 根据数据集版本设置设备特征
                if data in ['r5.2','r5.1','r6.2','r6.1']:
                    # 新版本增加了文件树长度特征
                    file_tree_len = len(df_acts_u.iloc[i]['content'].split(';'))
                    device_f = [connect_dur, file_tree_len]
                else:
                    device_f = [connect_dur]
                
            # ========== 3.3.5 检查是否为恶意活动 ==========
            is_mal_act = 0
            # 如果用户是恶意用户，且当前活动ID在恶意活动列表中
            if mal_u > 0 and df_acts_u.index[i] in users.loc[u]['malacts']:
                is_mal_act = 1

            # ========== 3.3.6 组装完整的特征向量 ==========
            # 特征向量组成：[用户ID, 天数, 活动类型, PC类型, 时间类型] + 各类活动特征 + [恶意活动标记, 内部威胁场景]
            oneu_week[i,:] = [ user_dict[u], time_convert(df_acts_u.iloc[i]['date'], 'dt2dn'), list_uacts_num[i], pc, act_time] \
            + device_f + file_f + http_f + email_f + [is_mal_act, mal_u]

            # 保存活动的元信息：[活动索引, PC ID, 时间戳]
            oneu_pc_time.append([df_acts_u.index[i], df_acts_u.iloc[i]['pc'], df_acts_u.iloc[i]['date']])
            
        # ========== 3.4 将当前用户的数据添加到总矩阵中 ==========
        u_week[current_ind:current_ind+len(oneu_week),:] = oneu_week
        pc_time += oneu_pc_time
        current_ind += len(oneu_week)
    
    # ========== 4. 数据后处理和保存 ==========
    # 截取实际使用的行数（去除初始化时的多余行）
    u_week = u_week[0:current_ind, :]
    
    # ========== 4.1 定义列名 ==========
    # 基础列：用户、天数、活动类型、PC类型、时间类型
    col_names = ['user','day','act','pc','time']
    
    # 根据数据集版本定义各类特征的列名
    if data in ['r4.1','r4.2']:
        device_feature_names = ['usb_dur']  # 设备特征：USB持续时间
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk', 'file_depth']
        http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords']
        email_feature_names = ['n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords']
    elif data in ['r5.2','r5.1', 'r6.2','r6.1']:
        device_feature_names = ['usb_dur', 'file_tree_len']  # 增加文件树长度
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk', 'file_depth', 'file_act', 'to_usb', 'from_usb']  # 增加USB传输特征
        http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords']
        if data in ['r6.2','r6.1']:
            http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords', 'http_act']  # r6版本增加HTTP活动类型
        # 邮件特征大幅增加，包含发送/接收标记和各种附件特征
        email_feature_names = ['send_mail', 'receive_mail','n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords']
        # 附件类型特征：压缩包、图片、文档、文本、可执行文件
        email_feature_names += ['e_att_other', 'e_att_comp', 'e_att_pho', 'e_att_doc', 'e_att_txt', 'e_att_exe']
        # 附件大小特征：对应各种类型附件的总大小
        email_feature_names += ['e_att_sother', 'e_att_scomp', 'e_att_spho', 'e_att_sdoc', 'e_att_stxt', 'e_att_sexe']     
        
    # 组合所有列名
    col_names = col_names + device_feature_names + file_feature_names+ http_feature_names + email_feature_names + ['mal_act','insider']
    
    # ========== 4.2 创建最终的DataFrame ==========
    # 包含元信息列和特征列
    df_u_week = pd.DataFrame(columns=['actid','pcid','time_stamp'] + col_names, index = np.arange(0,len(pc_time)))
    
    # 填充元信息
    df_u_week[['actid','pcid','time_stamp']] = np.array(pc_time)
    
    # 填充特征数据并转换为整数类型（除了时间戳）
    df_u_week[col_names] = u_week
    df_u_week[col_names] = df_u_week[col_names].astype(int)
    
    # ========== 4.3 保存处理结果 ==========
    # 将处理后的数值特征保存到NumDataByWeek文件夹
    # 这些文件将被后续的统计特征计算函数使用
    df_u_week.to_pickle("NumDataByWeek/"+str(week)+"_num.pickle")

##############################################################################

# return sessions for each user in a week:
# sessions[sid] = [sessionid, pc, start_with, end_with, start time, end time,number_of_concurent_login, [action_indices]]
# start_with: in the beginning of a week, action start with log in or not (1, 2)
# end_with: log off, next log on same computer (1, 2)
def get_sessions(uw, first_sid = 0):
    """
    从用户活动数据中提取会话信息
    参数:
    - uw: 用户活动数据
    - first_sid: 起始会话ID
    返回:
    - sessions: 会话字典,包含:
      - 会话ID
      - 电脑ID
      - 开始方式(1=登录开始,2=其他方式开始)
      - 结束方式(1=登出结束,2=下一次登录结束)
      - 开始时间
      - 结束时间
      - 并发登录数
      - 活动索引列表
    """
    sessions = {}
    open_sessions = {}
    sid = 0
    current_pc = uw.iloc[0]['pcid']
    start_time = uw.iloc[0]['time_stamp']
    if uw.iloc[0]['act'] == 1:
        open_sessions[current_pc] = [current_pc, 1, 0, start_time, start_time, 1, [uw.index[0]]]
    else:
        open_sessions[current_pc] = [current_pc, 2, 0, start_time, start_time, 1, [uw.index[0]]]

    for i in uw.index[1:]:
        current_pc = uw.loc[i]['pcid']
        if current_pc in open_sessions: # must be already a session with that pcid
            if uw.loc[i]['act'] == 2:
                open_sessions[current_pc][2] = 1
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
                sessions[sid] = [first_sid+sid] + open_sessions.pop(current_pc)
                sid +=1
            elif uw.loc[i]['act'] == 1:
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid+sid] + open_sessions.pop(current_pc)
                sid +=1
                #create a new open session
                open_sessions[current_pc] = [current_pc, 1, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
                if len(open_sessions) > 1: #increase the concurent count for all sessions
                    for k in open_sessions:
                        open_sessions[k][5] +=1
            else:
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
        else:
            start_status = 1 if uw.loc[i]['act'] == 1 else 2
            open_sessions[current_pc] = [current_pc, start_status, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
            if len(open_sessions) > 1: #increase the concurent count for all sessions
                for k in open_sessions:
                    open_sessions[k][5] +=1
    return sessions
                
def get_u_features_dicts(ul, data = 'r5.2'):
    """获取用户特征字典"""
    ufdict = {}
    list_uf=[] if data in ['r4.1','r4.2'] else ['project']
    list_uf += ['role','b_unit','f_unit', 'dept','team']
    for f in list_uf:
        ul[f] = ul[f].astype(str)
        tmp = list(set(ul[f]))
        tmp.sort()
        ufdict[f] = {idx:i for i, idx in enumerate(tmp)}
    return (ul,ufdict, list_uf)

def proc_u_features(uf, ufdict, list_f = None, data = 'r4.2'): #to remove mode
    """处理用户特征"""
    if type(list_f) != list:
        list_f=[] if data in ['r4.1','r4.2'] else ['project']
        list_f = ['role','b_unit','f_unit', 'dept','team'] + list_f

    out = []
    for f in list_f:
        out.append(ufdict[f][uf[f]])
    return out

def f_stats_calc(ud, fn, stats_f, countonly_f = {}, get_stats = False):
    """计算特征统计信息"""
    f_count = len(ud)
    r = []
    f_names = []
    
    for f in stats_f:
        inp = ud[f].values
        if get_stats:
            if f_count > 0:
                r += [np.min(inp), np.max(inp), np.median(inp), np.mean(inp), np.std(inp)]
            else: r += [0, 0, 0, 0, 0]
            f_names += [fn+'_min_'+f, fn+'_max_'+f, fn+'_med_'+f, fn+'_mean_'+f, fn+'_std_'+f]
        else:
            if f_count > 0: r += [np.mean(inp)]
            else: r += [0]
            f_names += [fn+'_mean_'+f]
        
    for f in countonly_f:
        for v in countonly_f[f]:
            r += [sum(ud[f].values == v)]
            f_names += [fn+'_n-'+f+str(v)]
    return (f_count, r, f_names)

def f_calc_subfeatures(ud, fname, filter_col, filter_vals, filter_names, sub_features, countonly_subfeatures):
    """计算子特征"""
    [n, stats, fnames] = f_stats_calc(ud, fname,sub_features, countonly_subfeatures)
    allf = [n] + stats
    allf_names = ['n_'+fname] + fnames
    for i in range(len(filter_vals)):
        [n_sf, sf_stats, sf_fnames] = f_stats_calc(ud[ud[filter_col]==filter_vals[i]], filter_names[i], sub_features, countonly_subfeatures)
        allf += [n_sf] + sf_stats
        allf_names += [fname+'_n_'+filter_names[i]] + [fname + '_' + x for x in sf_fnames]
    return (allf, allf_names)

def f_calc(ud, mode = 'week', data = 'r4.2'):
    """
    计算用户活动特征
    参数:
    - ud: 用户数据
    - mode: 计算模式(week/day/session)
    - data: 数据集名称
    功能:
    - 计算各类活动的统计特征
    - 区分工作时间/非工作时间特征
    - 生成特征向量
    """
    n_weekendact = (ud['time']==3).sum()
    if n_weekendact > 0: 
        is_weekend = 1
    else: 
        is_weekend = 0
    
    all_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    [all_f, all_f_names] = f_calc_subfeatures(ud, 'allact', None, [], [], [], all_countonlyf)
    if mode == 'day':
        [workhourf, workhourf_names] = f_calc_subfeatures(ud[(ud['time'] == 1) | (ud['time'] == 3)], 'workhourallact', None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(ud[(ud['time'] == 2) | (ud['time'] == 4) ], 'afterhourallact', None, [], [], [], all_countonlyf)
    elif mode == 'week':
        [workhourf, workhourf_names] = f_calc_subfeatures(ud[ud['time'] == 1], 'workhourallact', None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(ud[ud['time'] == 2 ], 'afterhourallact', None, [], [], [], all_countonlyf)
        [weekendf, weekendf_names] = f_calc_subfeatures(ud[ud['time'] >= 3 ], 'weekendallact', None, [], [], [], all_countonlyf)

    logon_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    logon_statf = []
        
    [all_logonf, all_logonf_names] = f_calc_subfeatures(ud[ud['act']==1], 'logon', None, [], [], logon_statf, logon_countonlyf)
    if mode == 'day':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & ((ud['time'] == 1) | (ud['time'] == 3) )], 'workhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & ((ud['time'] == 2) | (ud['time'] == 4) )], 'afterhourlogon', None, [], [], logon_statf, logon_countonlyf)
    elif mode == 'week':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] == 1)], 'workhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] == 2) ], 'afterhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [weekendlogonf, weekendlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] >= 3) ], 'weekendlogon', None, [], [], logon_statf, logon_countonlyf)
    
    device_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    device_statf = ['usb_dur','file_tree_len'] if data not in ['r4.1','r4.2'] else ['usb_dur']
        
    [all_devicef, all_devicef_names] = f_calc_subfeatures(ud[ud['act']==3], 'usb', None, [], [], device_statf, device_countonlyf)
    if mode == 'day':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & ((ud['time'] == 1) | (ud['time'] == 3) )], 'workhourusb', None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & ((ud['time'] == 2) | (ud['time'] == 4) )], 'afterhourusb', None, [], [], device_statf, device_countonlyf)
    elif mode == 'week':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] == 1)], 'workhourusb', None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] == 2) ], 'afterhourusb', None, [], [], device_statf, device_countonlyf)
        [weekenddevicef, weekenddevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] >= 3) ], 'weekendusb', None, [], [], device_statf, device_countonlyf)
          
    if mode != 'session': file_countonlyf = {'to_usb':[1],'from_usb':[1], 'file_act':[1,2,3,4], 'disk':[0,1], 'pc':[0,1,2,3]}
    else: file_countonlyf = {'to_usb':[1],'from_usb':[1], 'file_act':[1,2,3,4], 'disk':[0,1,2]}
    if data in ['r4.1','r4.2']: 
        [file_countonlyf.pop(k) for k in ['to_usb','from_usb', 'file_act']]
    
    (all_filef, all_filef_names) = f_calc_subfeatures(ud[ud['act']==7], 'file', 'file_type', [1,2,3,4,5,6], \
            ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
    
    if mode == 'day':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
    elif mode == 'week':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] ==1)], 'workhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] ==2)], 'afterhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (weekendfilef, weekendfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] >= 3)], 'weekendfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        
    email_stats_f = ['n_des', 'n_atts', 'n_exdes', 'n_bccdes', 'email_size', 'email_text_slen', 'email_text_nwords']
    if data not in ['r4.1','r4.2']:
        email_stats_f += ['e_att_other', 'e_att_comp', 'e_att_pho', 'e_att_doc', 'e_att_txt', 'e_att_exe']
        email_stats_f += ['e_att_sother', 'e_att_scomp', 'e_att_spho', 'e_att_sdoc', 'e_att_stxt', 'e_att_sexe'] 
        mail_filter = 'send_mail'
        mail_filter_vals = [0,1]
        mail_filter_names = ['recvmail','send_mail']
    else:
        mail_filter, mail_filter_vals, mail_filter_names = None, [], []    
    
    if mode != 'session': mail_countonlyf = {'Xemail':[1],'exbccmail':[1], 'pc':[0,1,2,3]}
    else: mail_countonlyf = {'Xemail':[1],'exbccmail':[1]}
    
    (all_emailf, all_emailf_names) = f_calc_subfeatures(ud[ud['act']==6], 'email', mail_filter, mail_filter_vals, mail_filter_names , email_stats_f, mail_countonlyf)
    if mode == 'week':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] == 1)], 'workhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] == 2)], 'afterhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (weekendemailf, weekendemailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] >= 3)], 'weekendemail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
    elif mode == 'day':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)    
    
    if data in ['r5.2','r5.1'] or data in ['r4.1','r4.2']:
        http_count_subf =  {'pc':[0,1,2,3]}
    elif data in ['r6.2','r6.1']:
        http_count_subf = {'pc':[0,1,2,3], 'http_act':[1,2,3]}
    
    if mode == 'session': http_count_subf.pop('pc',None)

    (all_httpf, all_httpf_names) = f_calc_subfeatures(ud[ud['act']==5], 'http', 'http_type', [1,2,3,4,5,6], \
            ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
    
    if mode == 'week':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] ==1)], 'workhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] ==2)], 'afterhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (weekendhttpf, weekendhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] >=3)], 'weekendhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
    elif mode == 'day':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        
    numActs = all_f[0]
    mal_u = 0
    if (ud['mal_act']).sum() > 0:
        tmp = list(set(ud['insider']))
        if len(tmp) > 1:
            tmp.remove(0.0)
        mal_u = tmp[0]
        
    if mode == 'week':        
        features_tmp =  all_f + workhourf + afterhourf + weekendf +\
                        all_logonf + workhourlogonf + afterhourlogonf + weekendlogonf +\
                        all_devicef + workhourdevicef + afterhourdevicef + weekenddevicef +\
                        all_filef + workhourfilef + afterhourfilef + weekendfilef + \
                        all_emailf + workhouremailf + afterhouremailf + weekendemailf + all_httpf + workhourhttpf + afterhourhttpf + weekendhttpf
        fnames_tmp = all_f_names + workhourf_names + afterhourf_names + weekendf_names +\
                      all_logonf_names + workhourlogonf_names + afterhourlogonf_names + weekendlogonf_names +\
                      all_devicef_names + workhourdevicef_names + afterhourdevicef_names + weekenddevicef_names +\
                      all_filef_names + workhourfilef_names + afterhourfilef_names + weekendfilef_names + \
                      all_emailf_names + workhouremailf_names + afterhouremailf_names + weekendemailf_names + all_httpf_names + workhourhttpf_names + afterhourhttpf_names + weekendhttpf_names
    elif mode == 'day':
        features_tmp = all_f + workhourf + afterhourf +\
                        all_logonf + workhourlogonf + afterhourlogonf +\
                        all_devicef + workhourdevicef + afterhourdevicef + \
                        all_filef + workhourfilef + afterhourfilef + \
                        all_emailf + workhouremailf + afterhouremailf + all_httpf + workhourhttpf + afterhourhttpf
        fnames_tmp = all_f_names + workhourf_names + afterhourf_names +\
                      all_logonf_names + workhourlogonf_names + afterhourlogonf_names +\
                      all_devicef_names + workhourdevicef_names + afterhourdevicef_names +\
                      all_filef_names + workhourfilef_names + afterhourfilef_names + \
                      all_emailf_names + workhouremailf_names + afterhouremailf_names + all_httpf_names + workhourhttpf_names + afterhourhttpf_names
    elif mode == 'session':
        features_tmp = all_f + all_logonf + all_devicef + all_filef + all_emailf + all_httpf
        fnames_tmp = all_f_names + all_logonf_names + all_devicef_names + all_filef_names + all_emailf_names + all_httpf_names
    
    return [numActs, is_weekend, features_tmp, fnames_tmp, mal_u]

def session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf):
    """
    计算会话实例的特征
    参数:
    - ud: 用户数据
    - sinfo: 会话信息
    - week: 周数
    - mode: 模式(week/day/session)
    - data: 数据集名称
    - uw: 用户周数据
    - v: 用户ID
    - list_uf: 用户特征列表
    返回:
    - 会话特征向量和特征名称
    功能:
    - 计算会话的时间特征(工作时间比例,持续时间等)
    - 计算会话的活动特征
    - 合并用户特征和会话特征
    """
    d = ud.iloc[0]['day']
    perworkhour = sum(ud['time']==1)/len(ud)
    perafterhour = sum(ud['time']==2)/len(ud)
    perweekend = sum(ud['time']==3)/len(ud)
    perweekendafterhour = sum(ud['time']==4)/len(ud)
    st_timestamp = min(ud['time_stamp'])
    end_timestamp = max(ud['time_stamp'])
    s_dur = (end_timestamp - st_timestamp).total_seconds() / 60 # in minute
    s_start = st_timestamp.hour + st_timestamp.minute/60
    s_end = end_timestamp.hour + end_timestamp.minute/60
    starttime = st_timestamp.timestamp()
    endtime = end_timestamp.timestamp()
    n_days = len(set(ud['day']))        
    
    tmp = f_calc(ud, mode, data)
    session_instance = [starttime, endtime, v, sinfo[0], d, week, ud.iloc[0]['pc'], perworkhour, perafterhour, perweekend,
                        perweekendafterhour, n_days, s_dur, sinfo[6], sinfo[2], sinfo[3], s_start, s_end] + \
        (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [tmp[4]]
    return (session_instance, tmp[3])

def to_csv(week, mode, data, ul, uf_dict, list_uf, subsession_mode = {}):
    """
    将处理后的数据导出为CSV格式
    参数:
    - week: 周数
    - mode: 模式(week/day/session)
    - data: 数据集名称
    - ul: 用户列表
    - uf_dict: 用户特征字典
    - list_uf: 用户特征列表
    - subsession_mode: 子会话模式配置
    功能:
    - 根据不同模式(周/日/会话)提取特征
    - 处理子会话(如果需要)
    - 将特征数据保存为pickle文件,最终合并为CSV
    - 支持按时间(time)或活动数量(nact)划分子会话
    """
    user_dict = {i : idx for (i, idx) in enumerate(ul.index)} 
    if mode == 'session': 
        first_sid = week*100000 # to get an unique index for each session, also, first 1 or 2 number in index would be week number
        cols2a = ['starttime', 'endtime','user', 'sessionid', 'day', 'week', 'pc', 'isworkhour', 'isafterhour','isweekend', 
                  'isweekendafterhour', 'n_days', 'duration', 'n_concurrent_sessions', 'start_with', 'end_with', 'ses_start', 
                  'ses_end'] + list_uf + ['ITAdmin','O','C','E','A','N']
    elif mode == 'day': 
        cols2a = ['starttime', 'endtime','user', 'day', 'week', 'isweekday','isweekend'] + list_uf +\
            ['ITAdmin','O','C','E','A','N']
    else: cols2a = ['starttime', 'endtime','user','week'] + list_uf + ['ITAdmin','O','C','E','A','N']
    cols2b = ['insider']        

    w = pd.read_pickle("NumDataByWeek/"+str(week)+"_num.pickle")

    usnlist = list(set(w['user'].astype('int').values))
    if True:
        cols = ['week']+ list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'] 
        uw = pd.DataFrame(columns = cols, index = user_dict.keys())
        uwdict = {}
        for v in user_dict:
            if v in usnlist:
                is_ITAdmin = 1 if ul.loc[user_dict[v], 'role'] == 'ITAdmin' else 0
                row = [week] + proc_u_features(ul.loc[user_dict[v]], uf_dict, list_uf, data = data) + [is_ITAdmin] + \
                    (ul.loc[user_dict[v],['O','C','E','A','N']]).tolist() + [0]
                row[-1] = int(list(set(w[w['user']==v]['insider']))[0])
                uwdict[v] = row
        uw = pd.DataFrame.from_dict(uwdict, orient = 'index',columns = cols)    
    
    towrite = pd.DataFrame()
    towrite_list = []
    
    if mode == 'session' and len(subsession_mode) > 0:
        towrite_list_subsession = {} 
        for k1 in subsession_mode:
            towrite_list_subsession[k1] = {}
            for k2 in subsession_mode[k1]:
                towrite_list_subsession[k1][k2] = []
    
    days = list(set(w['day']))
    for v in user_dict:
        if v in usnlist:
            uactw = w[w['user']==v]
            
            if mode == 'week':
                a = uactw.iloc[0]['time_stamp']
                a = a - timedelta(int(a.strftime("%w"))) # get the nearest Sunday
                starttime = datetime(a.year, a.month, a.day).timestamp()
                endtime = (datetime(a.year, a.month, a.day) + timedelta(days=7)).timestamp()
                
                if len(uactw) > 0:
                    tmp = f_calc(uactw, mode, data)
                    i_fnames = tmp[3]
                    towrite_list.append([starttime, endtime, v, week] + (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [ tmp[4]])

            if mode == 'session':
                sessions = get_sessions(uactw, first_sid)
                first_sid += len(sessions)
                for s in sessions:
                    sinfo = sessions[s]
                    
                    ud = uactw.loc[sessions[s][7]]
                    if len(ud) > 0:                     
                        session_instance, i_fnames = session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf)
                        towrite_list.append(session_instance)
                        
                        ## do subsessions:
                        if 'time' in subsession_mode: # divide a session into subsessions by consecutive time chunks
                            for subsession_dur in subsession_mode['time']:
                                n_subsession = int(np.ceil(session_instance[12] / subsession_dur))
                                if n_subsession == 1:
                                    towrite_list_subsession['time'][subsession_dur].append([0] + session_instance)
                                else:
                                    sinfo1 = sinfo.copy()
                                    for subsession_ind in range(n_subsession):
                                        sinfo1[3] = 0 if subsession_ind < n_subsession-1 else sinfo[3] 
                                        
                                        subsession_ud = ud[(ud['time_stamp'] >= sessions[s][4] + timedelta(minutes = subsession_ind*subsession_dur)) & \
                                                            (ud['time_stamp'] < sessions[s][4] + timedelta(minutes = (subsession_ind+1)*subsession_dur))]
                                        if len(subsession_ud) > 0:
                                            ss_instance, _ = session_instance_calc(subsession_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                            towrite_list_subsession['time'][subsession_dur].append([subsession_ind] + ss_instance)
                            
                        if 'nact' in subsession_mode:
                            for ss_nact in subsession_mode['nact']:
                                n_subsession = int(np.ceil(len(ud) / ss_nact))
                                if n_subsession == 1:
                                    towrite_list_subsession['nact'][ss_nact].append([0] + session_instance)
                                else:
                                    sinfo1 = sinfo.copy()
                                    for ss_ind in range(n_subsession):
                                        sinfo1[3] = 0 if ss_ind < n_subsession-1 else sinfo[3] 
                                        
                                        ss_ud = ud.iloc[ss_ind*ss_nact : min(len(ud), (ss_ind+1)*ss_nact)] 
                                        if len(ss_ud) > 0:
                                            ss_instance,_ = session_instance_calc(ss_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                            towrite_list_subsession['nact'][ss_nact].append([ss_ind] + ss_instance)
                        
            if mode == 'day':
                days = sorted(list(set(uactw['day']))) 
                for d in days:
                    ud = uactw[uactw['day'] == d]
                    isweekday = 1 if sum(ud['time']>=3) == 0 else 0
                    isweekend = 1-isweekday
                    a = ud.iloc[0]['time_stamp']
                    starttime = datetime(a.year, a.month, a.day).timestamp()
                    endtime = (datetime(a.year, a.month, a.day) + timedelta(days=1)).timestamp()
                    
                    if len(ud) > 0:
                        tmp = f_calc(ud, mode, data)
                        i_fnames = tmp[3]
                        towrite_list.append([starttime, endtime, v, d, week, isweekday, isweekend] + (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [ tmp[4]])

    towrite = pd.DataFrame(columns = cols2a + i_fnames + cols2b, data = towrite_list)
    towrite.to_pickle("tmp/"+str(week) + mode+".pickle")
    
    if mode == 'session' and len(subsession_mode) > 0:
        for k1 in subsession_mode:
            for k2 in subsession_mode[k1]:
                df_tmp = pd.DataFrame(columns = ['subs_ind']+cols2a + i_fnames + cols2b, data = towrite_list_subsession[k1][k2])
                df_tmp.to_pickle("tmp/"+str(week) + mode + k1 + str(k2) + ".pickle")
    
if __name__ == "__main__":
    """
    CERT内部威胁数据集特征提取主程序
    
    这个主程序实现了完整的CERT数据集特征提取流水线，包括：
    1. 原始日志数据的按周合并
    2. 用户信息和恶意用户标记的提取
    3. 活动数据的数值化特征提取
    4. 多粒度特征的统计计算和CSV导出
    
    最终输出多种格式的特征文件，支持周级别、日级别和会话级别的分析
    """
    
    # ==================== 第一部分：环境检查与初始化 ====================
    """
    环境检查与初始化阶段：
    - 验证脚本运行环境是否为有效的CERT数据集目录
    - 创建必要的临时目录用于存储中间处理结果
    - 确保数据处理的文件结构正确性
    """
    
    # 获取当前工作目录的名称，应该是CERT数据集的版本号（如r4.2, r5.2等）
    dname = os.getcwd().split('/')[-1]
    
    # 验证目录名是否为支持的CERT数据集版本
    # 目前支持的版本：r4.1, r4.2, r5.1, r5.2, r6.1, r6.2
    if dname not in ['r4.1','r4.2','r6.2','r6.1','r5.1','r5.2']:
        raise Exception('Please put this script in and run it from a CERT data folder (e.g. r4.2)')
    
    # 创建数据处理流水线所需的临时目录
    # tmp: 存储临时的pickle文件和中间处理结果
    # ExtractedData: 存储最终的CSV格式特征文件
    # DataByWeek: 存储按周合并的原始活动数据
    # NumDataByWeek: 存储数值化后的周活动特征
    for folder_name in ["tmp", "ExtractedData", "DataByWeek", "NumDataByWeek"]:
        os.makedirs(folder_name, exist_ok=True)
    
    # ==================== 第二部分：配置参数设置 ====================
    """
    参数配置阶段：
    - 设置子会话划分策略（可选功能）
    - 配置并行处理的CPU核心数
    - 确定数据集的总周数
    """
    
    # 子会话模式配置：用于将长会话分割为更小的子会话进行分析
    # 'nact': 按活动数量划分，[25, 50]表示生成25个活动一组和50个活动一组的子会话
    # 'time': 按时间长度划分，[120, 240]表示生成120分钟和240分钟的子会话
    # 如果不需要子会话分析，可以设置为空字典 {}
    subsession_mode = {'nact':[25, 50], 'time':[120, 240]}
    
    # 并行处理配置：默认使用8个CPU核心进行并行计算
    numCores = 8
    # 支持通过命令行参数自定义核心数，例如：python feature_extraction.py 16
    arguments = len(sys.argv) - 1
    if arguments > 0:
        numCores = int(sys.argv[1])
        
    # 确定数据集的总周数：不同版本的CERT数据集包含的周数不同
    # r4.x版本：73周，其他版本（r5.x, r6.x）：75周
    numWeek = 73 if dname in ['r4.1','r4.2'] else 75
    
    # 初始化计时器，用于跟踪各阶段的处理时间
    st = time.time()
    
    # ==================== Step 1: 按周合并原始数据 ====================
    """
    第一步：原始数据按周合并
    
    功能说明：
    - 读取原始的CSV日志文件（http.csv, email.csv, file.csv, logon.csv, device.csv）
    - 根据时间戳将所有活动数据按周进行分组和合并
    - 处理不同数据集版本间的格式差异
    - 为每条活动记录添加统一的type标识符
    
    输入文件：
    - http.csv: HTTP访问日志
    - email.csv: 邮件活动日志  
    - file.csv: 文件操作日志
    - logon.csv: 登录/登出日志
    - device.csv: 设备连接日志
    
    输出：
    - DataByWeek/0.pickle, DataByWeek/1.pickle, ...: 每周的合并活动数据
    - 每个文件包含该周所有类型的用户活动，按时间排序
    """
    print("Step 1: 开始按周合并原始数据...")
    combine_by_timerange_pandas(dname)
    print(f"Step 1 - 按周分离数据完成. 耗时 (分钟): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    # ==================== Step 2: 获取用户信息和恶意用户标记 ====================
    """
    第二步：用户信息提取和恶意用户标记
    
    功能说明：
    - 从LDAP目录读取所有用户的组织信息（部门、角色、项目等）
    - 处理用户的入职/离职时间信息
    - 读取心理测量数据（如果存在）
    - 从answers目录读取恶意用户的真实标签和活动详情
    - 确定每个用户的主要工作PC和共享PC
    - 标记恶意活动的时间窗口和具体活动ID
    
    输入文件：
    - LDAP/*.csv: 用户组织信息（按月）
    - psychometric.csv: 心理测量数据（可选）
    - answers/insiders.csv: 内部威胁者列表
    - answers/r*.*/: 具体的恶意活动详情
    
    输出：
    - 用户信息DataFrame，包含：
      - 基础信息：姓名、邮箱、角色、部门等
      - PC信息：主要PC、共享PC列表
      - 恶意标记：是否为内部威胁者、恶意活动时间窗口、具体恶意活动ID列表
      - 心理特征：大五人格特征分数（如果有数据）
    """
    print("Step 2: 开始获取用户列表和恶意用户信息...")
    users = get_mal_userdata(dname)
    print(f"Step 2 - 获取用户列表完成. 耗时 (分钟): {(time.time()-st)/60:.2f}")
    print(f"总用户数: {len(users)}, 恶意用户数: {len(users[users['malscene'] > 0])}")
    st = time.time()
    
    # ==================== Step 3: 活动数据数值化特征提取 ====================
    """
    第三步：将活动数据转换为数值特征
    
    功能说明：
    - 并行处理每周的活动数据，将原始文本日志转换为数值特征向量
    - 为每个活动提取多维特征：时间特征、PC特征、内容特征等
    - 根据不同活动类型应用专门的特征提取函数
    - 标记恶意活动和正常活动
    - 处理设备连接的时间配对逻辑
    
    处理过程：
    - 读取每周的合并数据（从DataByWeek/）
    - 逐用户、逐活动进行特征提取
    - 活动类型映射：logon(1), logoff(2), connect(3), disconnect(4), http(5), email(6), file(7)
    - 时间类型分类：工作日工作时间(1), 工作日非工作时间(2), 周末工作时间(3), 周末非工作时间(4)
    - PC类型分类：自己的PC(0), 共享PC(1), 他人PC(2), 主管PC(3)
    
    特征维度（根据数据集版本）：
    - r4.x: 27维特征
    - r5.x: 45维特征 
    - r6.x: 46维特征
    
    输出：
    - NumDataByWeek/0_num.pickle, NumDataByWeek/1_num.pickle, ...: 每周的数值化特征
    - 每行代表一个活动，包含用户ID、时间、活动类型、各类特征值、恶意标记等
    """
    print("Step 3: 开始将活动转换为数值化特征...")
    print(f"使用 {numCores} 个CPU核心进行并行处理...")
    
    # 使用joblib.Parallel进行并行处理，显著提升大数据集的处理速度
    # delayed()将函数调用包装为延迟执行的任务
    # n_jobs指定并行进程数，-1表示使用所有可用CPU核心
    Parallel(n_jobs=numCores)(delayed(process_week_num)(i, users, data=dname) for i in range(numWeek))
    
    print(f"Step 3 - 活动数值化转换完成. 耗时 (分钟): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    # ==================== Step 4: 多粒度特征统计和CSV导出 ====================
    """
    第四步：多粒度特征提取和CSV格式导出
    
    功能说明：
    - 从数值化的活动数据中计算更高级别的统计特征
    - 支持三种不同的时间粒度：周级别、日级别、会话级别
    - 为每种粒度计算丰富的统计特征（计数、均值、方差等）
    - 区分工作时间、非工作时间、周末的活动模式
    - 合并用户属性特征和行为特征
    - 导出为CSV格式，便于后续的机器学习分析
    
    三种分析粒度：
    1. 周级别 (week): 以周为单位聚合用户行为，适合长期趋势分析
    2. 日级别 (day): 以天为单位聚合用户行为，适合日常模式分析  
    3. 会话级别 (session): 以登录会话为单位，适合细粒度行为分析
    
    特征类型：
    - 活动计数特征：各类活动的数量统计
    - 时间分布特征：工作时间vs非工作时间的活动分布
    - 内容特征统计：文件大小、邮件长度、URL长度等的统计值
    - PC使用模式：不同类型PC的使用频率
    - 用户属性特征：部门、角色、心理特征等
    """
    print("Step 4: 开始多粒度特征提取和CSV导出...")
    
    # 处理三种不同的时间粒度
    for mode in ['week','day','session']:
        print(f"正在处理 {mode} 级别的特征...")
        
        # 确定每种模式需要处理的周范围
        # 周级别从第1周开始（第0周用于确定用户PC），日级别和会话级别从第0周开始
        weekRange = list(range(0, numWeek)) if mode in ['day', 'session'] else list(range(1, numWeek))
        print(f"{mode} 模式处理周范围: {weekRange[0]} 到 {weekRange[-1]} (共 {len(weekRange)} 周)")
        
        # 获取用户特征的字典映射和列表
        # ul: 处理后的用户DataFrame，uf_dict: 特征值到数值的映射字典，list_uf: 特征列名列表
        (ul, uf_dict, list_uf) = get_u_features_dicts(users, data=dname)
        print(f"用户特征维度: {len(list_uf)}, 包含特征: {list_uf}")
        
        # 并行处理每一周的数据，计算该周的特征并保存为临时pickle文件
        print(f"开始并行处理 {len(weekRange)} 周的数据...")
        Parallel(n_jobs=numCores)(delayed(to_csv)(i, mode, dname, ul, uf_dict, list_uf, subsession_mode) 
                                   for i in weekRange)
        
        # ==================== 合并所有周的数据为单一CSV文件 ====================
        print(f"开始合并 {mode} 模式的所有周数据为CSV文件...")
        
        # 创建输出CSV文件，文件名格式：mode + 数据集版本.csv (如weekr4.2.csv)
        csv_filename = f'ExtractedData/{mode}{dname}.csv'
        all_csv = open(csv_filename, 'a')
        
        # 读取第一周的数据并写入CSV文件头
        first_week_file = f"tmp/{weekRange[0]}{mode}.pickle"
        towrite = pd.read_pickle(first_week_file)
        towrite.to_csv(all_csv, header=True, index=False)
        print(f"第一周数据写入完成，特征维度: {towrite.shape[1]}, 样本数: {towrite.shape[0]}")
        
        # 逐一读取剩余周的数据并追加到CSV文件（不包含文件头）
        total_samples = towrite.shape[0]
        for w in weekRange[1:]:
            week_file = f"tmp/{w}{mode}.pickle"
            towrite = pd.read_pickle(week_file)        
            towrite.to_csv(all_csv, header=False, index=False)
            total_samples += towrite.shape[0]
            
        all_csv.close()
        print(f"{mode} 模式主文件完成: {csv_filename}, 总样本数: {total_samples}")
        
        # ==================== 处理子会话数据（仅限会话模式） ====================
        if mode == 'session' and len(subsession_mode) > 0:
            print("开始处理子会话数据...")
            
            # 遍历所有子会话配置
            for k1 in subsession_mode:  # k1: 'nact' 或 'time'
                for k2 in subsession_mode[k1]:  # k2: 具体的数值（如25, 50, 120, 240）
                    print(f"处理子会话类型: {k1}, 参数: {k2}")
                    
                    # 创建子会话CSV文件，文件名格式：session + 类型 + 参数 + 数据集.csv
                    # 例如：sessionnact25r4.2.csv, sessiontime120r4.2.csv
                    subsession_csv_filename = f'ExtractedData/{mode}{k1}{k2}{dname}.csv'
                    all_csv = open(subsession_csv_filename, 'a')
                    
                    # 读取第一周的子会话数据
                    first_subsession_file = f'tmp/{weekRange[0]}{mode}{k1}{k2}.pickle'
                    towrite = pd.read_pickle(first_subsession_file)
                    towrite.to_csv(all_csv, header=True, index=False)
                    
                    # 追加剩余周的子会话数据
                    subsession_samples = towrite.shape[0]
                    for w in weekRange[1:]:
                        subsession_file = f'tmp/{w}{mode}{k1}{k2}.pickle'
                        towrite = pd.read_pickle(subsession_file)        
                        towrite.to_csv(all_csv, header=False, index=False)
                        subsession_samples += towrite.shape[0]
                        
                    all_csv.close()
                    print(f"子会话文件完成: {subsession_csv_filename}, 样本数: {subsession_samples}")
        
        # 打印当前模式完成信息和耗时
        print(f'{mode} 模式数据提取完成. 耗时 (分钟): {(time.time()-st)/60:.2f}')
        st = time.time()

    # ==================== 第五部分：清理临时文件 ====================
    """
    清理阶段：
    - 删除处理过程中产生的临时目录和文件
    - 保留ExtractedData目录中的最终CSV输出文件
    - 释放磁盘空间，保持工作目录整洁
    
    被删除的目录：
    - tmp/: 临时pickle文件
    - DataByWeek/: 按周合并的原始数据
    - NumDataByWeek/: 数值化的周特征数据
    
    保留的目录：
    - ExtractedData/: 最终的CSV特征文件
    """
    print("开始清理临时文件...")
    cleanup_dirs = ["tmp", "DataByWeek", "NumDataByWeek"]
    for dir_name in cleanup_dirs:
        os.system(f"rm -r {dir_name}")
        print(f"已删除临时目录: {dir_name}")
    
    print("="*60)
    print("CERT数据集特征提取流水线执行完成！")
    print("最终输出文件位于 ExtractedData/ 目录:")
    print(f"- 周级别特征: ExtractedData/week{dname}.csv")
    print(f"- 日级别特征: ExtractedData/day{dname}.csv") 
    print(f"- 会话级别特征: ExtractedData/session{dname}.csv")
    if len(subsession_mode) > 0:
        print("- 子会话特征文件:")
        for k1 in subsession_mode:
            for k2 in subsession_mode[k1]:
                print(f"  - ExtractedData/session{k1}{k2}{dname}.csv")
    print("="*60)
