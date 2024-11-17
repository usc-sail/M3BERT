# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ corr_analysis.py ]
#   Synopsis     [ correlation analysis for the mtg dataset ]
#   Author       [ Timothy D. Greer (timothydgreer) ]
#   Copyright    [ Copyleft(c), SAIL Lab, USC, USA ]
"""*********************************************************************************************"""
import numpy as np
import torch
import glob as glob
from scipy.stats import pearsonr
import csv

files = glob.glob('./mtg_extracted/*')
files_phen = glob.glob('comp/*')
phen_dir = ''.join(files_phen[0].split('/')[:-1])
files_phen = [i.split('/')[-1] for i in files_phen]
for i in files:
    #print(i.split('/')[-1])
    #print(files_phen)
    if i.split('/')[-1] in files_phen:
        print("here")
        print(phen_dir)
        print(i)
        phen = np.load(phen_dir+'/'+i.split('/')[-1])
        feats = np.load(i)
        print(phen.shape)
        for j in range(feats.shape[0]):
            for k in range(feats.shape[2]):
                for f in range(phen.shape[1]):
                    temp_feats = feats[j,:,k]
                    temp_phen = phen[:,f]
                    #temp_phen = np.exp(temp_phen[:len(temp_feats)])
                    #if np.isnan(temp_phen).any() or np.isinf(temp_phen).any():
                    #    continue                   
                    #print("here") 
                    if pearsonr(temp_feats, temp_phen[:len(temp_feats)])[0] > .6:
                        if f not in [1000]:#[2,3,6,9]:
                            #print("This is feature: ")
                            #print(f)
                            
                            print("Correlation is: ")
                            print(pearsonr(temp_feats, temp_phen[:len(temp_feats)])[1])
                            print(j)
                            with open(str(f)+'_'+str(j)+'_'+str(k)+'_'+'feat.csv','w') as myfile:
                                write = csv.writer(myfile)
                                print(temp_feats)
                                write.writerows([[x] for x in temp_feats])
                                myfile.close()
                            with open(str(f)+'_'+str(j)+'_'+str(k)+'_'+'phen.csv','w') as h:
                                writer = csv.writer(h)
                                print(temp_phen)
                                writer.writerows([[x] for x in temp_phen])
                                h.close()
                            
       
