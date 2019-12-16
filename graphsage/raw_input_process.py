#!/data/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas  as pd
import json
import networkx as nx
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import codecs
import io
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

cloumns_name=['campany_id','name','cate_2','established','reg_status','reg_capital','employee_num','phone_number','property1','put_reason_num','abnormal_score','executives_score','check_score','illegal_score','revoke_license','chattel_mortgage','stock_pledge','judicial_assistance','address_change','reg_capital_change','legal_person_change','shareholder_change','reg_status_change','risk_lawsuit','criminal_case','limit_consumption','dishonest','debetor','own_tax','new_own_tax','grade','employment','total_null_ratio','risk_null_ratio','ds']



def main(arg):
    date_name = arg[1]
    campany_id_label_train= pd.read_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/campany_train.txt".format(date_name),sep='	',engine='python')
    campany_relation= pd.read_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/campany_relation.txt".format(date_name),sep='	',engine='python')
    feature_all= pd.read_csv("/data/radeliu/ceph/radeliu/campany_relation/e_feature/{:s}/ds={:s}/000000_0.gz".format(date_name,date_name),compression='gzip',sep='|',names=cloumns_name,error_bad_lines=False)
    #feature_mini= pd.read_csv("/data/radeliu/ceph/radeliu/campany_relation/feature_all.txt",sep='	',error_bad_lines=False,engine='python',encoding='utf-8')

    feature_mini=feature_all

    #feature_mini.columns=feature_mini.columns.map(lambda x:x.split('.')[1])
    #feature_mini.campany_id.to_csv("/data/radeliu/ceph/radeliu/campany_relation/graph_data/campany_id_from_feature.txt",index=False,header=True)


    #result=[]
    #fin = io.open('/data/radeliu/ceph/radeliu/campany_relation/feature_mini.txt', 'r',encoding='utf-8')
    #for eachLine in fin:
    #    line = eachLine.strip().encode('utf-8', 'ignore').decode('utf-8')
    #    result.append(line.split('	'))


    all_id=list(set(list(campany_relation.campany1)+list(campany_relation.campany2)))

    id_map={ all_id[i]:i for i in range(0,len(all_id))}

    source_map=campany_relation.campany1.map(id_map)
    target_map=campany_relation.campany2.map(id_map)
    links=pd.concat([source_map,target_map],axis=1)
    links_revers=links[['campany2','campany1']]
    all_links=pd.concat([pd.DataFrame(links.values),pd.DataFrame(links_revers.values)],ignore_index=True)
    all_links.columns=['campany1','campany2']
    all_links.sort_values(by='campany1',inplace=True)
    all_links_groupby = all_links.groupby('campany1')['campany2'].apply(list).reset_index(name='campany2')
    graph_initial=all_links_groupby.campany2.to_dict()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_initial))

    campanyid_label=campany_id_label_train.campanyid.map(id_map)
    campanyid_label_for_train_map=pd.concat([campanyid_label,campany_id_label_train.label],axis=1)

    campanyid_label_for_train_filter=campanyid_label_for_train_map.drop_duplicates(subset='campanyid', keep="last")
    #campanyid_label_for_train_filter=campanyid_label_for_train.ix[~((campanyid_label_for_train.campanyid.isin([4093809,307371,3933209,3116858,2711809])==True)&(campanyid_label_for_train.label==0)),:]
    train, test = train_test_split(campanyid_label_for_train_filter, test_size=0.3)

    val_index=pd.Series(list(set(all_id).difference(set(campany_id_label_train.campanyid.values)))).map(id_map).values


    print("all_id length:",len(all_id))
    print("campanyid_label_for_train_filter:",(len(campanyid_label_for_train_filter),campanyid_label_for_train_filter.campanyid.nunique()))



    #feature_mini.columns=feature_mini.columns.map(lambda x:x.split('.')[1])
    id_lack=set(set(all_id)).difference(list(feature_mini.campany_id.values))
    random_id=np.random.choice(range(len(feature_mini)), len(id_lack), replace=False)
    id_lack_feature=pd.concat([pd.DataFrame(list(id_lack),columns=['campany_id']),feature_mini.ix[random_id,1:].reset_index()],axis=1)
    feature_mini=pd.concat([feature_mini,id_lack_feature],ignore_index=True)


    remove_columns=['name','phone_number','property1','ds','campany_id','cate_2']#,'put_reason_num'
    select_columns=['campany_id','cate_2']+[column for column in feature_mini.columns if column not in remove_columns]

    feature_mini_select=feature_mini[select_columns]
    feature_mini_new=feature_mini_select
    print("name0:",feature_mini_new.columns)

    put_reason_num_process=feature_mini_new['put_reason_num'].fillna('无:0').str.split(':',expand=True).add_prefix('put_reason_num')
    print("name1:",put_reason_num_process.columns)
    feature_mini_new=pd.concat([feature_mini_new,put_reason_num_process['put_reason_num1']],axis=1)
    feature_mini_new.drop('put_reason_num',axis=1,inplace=True)
    print("name2:",feature_mini_new.columns)
    feature_mini_new.put_reason_num1=feature_mini_new.put_reason_num1.fillna(0).astype(int)
    feature_mini_new.cate_2 = feature_mini_new.cate_2.astype('category').cat.codes
    feature_mini_new.grade = feature_mini_new.grade.astype('category').cat.codes
    print("name3:",feature_mini_new.columns)
    #feature_mini_new_one_hot=pd.get_dummies(feature_mini_new)
    feature_mini_new_one_hot=feature_mini_new
    feature_mini_new_one_hot=feature_mini_new_one_hot.fillna(0)
    feature_mini_new_one_hot.replace('\\N',0,inplace=True)
    #feature_mini_new_one_hot.replace('A',0,inplace=True)
    print("dtypes:",feature_mini_new_one_hot.dtypes)
    feature_mini_new_one_hot_scale = pd.concat([feature_mini_new_one_hot[['campany_id','cate_2']],pd.DataFrame(preprocessing.scale(feature_mini_new_one_hot.iloc[:,2:]),columns=feature_mini_new_one_hot.columns[2:])],axis=1).round(2)
    #print("name2:",feature_mini_new.columns)
    #print("past1")
    #feature_mini_new_one_hot_scale = feature_mini_new.fillna(0)
    #print("past2")
    #print("name3:",feature_mini_new_one_hot_scale.columns)
    #print("1:",feature_mini_new_one_hot_scale.campany_id.map(id_map).name)
    #print("2:",feature_mini_new_one_hot_scale.ix[:,1:].columns)

    #print('length:',len(feature_mini_new_one_hot_scale))
    #print("past5")
    #feature_mini_new_one_hot_scale=feature_mini_new_one_hot
    feature_mini_new_one_hot_scale_map=pd.concat([feature_mini_new_one_hot_scale.campany_id.map(id_map),feature_mini_new_one_hot_scale.ix[:,1:]],axis=1).sort_values(by='campany_id')
    #print("past3")

    train_data = pd.concat([feature_mini_new_one_hot_scale_map.ix[train.campanyid, 1:].reset_index(drop=True), train.label.reset_index(drop=True)], axis=1)
    test_data = pd.concat([feature_mini_new_one_hot_scale_map.ix[test.campanyid, 1:].reset_index(drop=True), test.label.reset_index(drop=True)], axis=1)


    print("train:",(len(train),train.campanyid.nunique()))
    print("test:",(len(test),test.campanyid.nunique()))
    print("val:",len(val_index))

    print('feature',feature_mini_new_one_hot_scale_map.shape)
    print('adj',adj.shape)


    train_data.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/train_data.txt".format(date_name), index=False, header=False)
    test_data.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/test_data.txt".format(date_name), index=False, header=False)

    sparse.save_npz('/data/radeliu/ceph/radeliu/campany_relation/{:s}/adj.npz'.format(date_name), adj)

    train.campanyid.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/train_index.txt".format(date_name),index=False,header=False)
    train.label.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/train_y.txt".format(date_name),index=False,header=False)
    test.campanyid.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/val_index.txt".format(date_name),index=False,header=False)
    test.label.to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/val_y.txt".format(date_name),index=False,header=False)

    pd.DataFrame(val_index).to_csv("/data/radeliu/ceph/radeliu/campany_relation/{:s}/test_index.txt".format(date_name),index=False,header=False)
    print("past4")
    np.savez('/data/radeliu/ceph/radeliu/campany_relation/{:s}/feature.npz'.format(date_name),feature_mini_new_one_hot_scale_map.ix[:,1:].values)

    print('feature', feature_mini_new_one_hot_scale_map.shape)
    print('adj', adj.shape)
    print("pastall")

if __name__ == "__main__":
    main(sys.argv)