# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner_m3bert.py ]
#   Synopsis     [ runner for the m3bert model ]
#   Author       [ Timothy D. Greer (timothydgreer) ]
#   Copyright    [ Copyleft(c), SAIL Lab, USC, USA ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import random
import torch
import numpy as np


############
# CONSTANT #
############
DR = 3
HIDDEN_SIZE = 768
MASK_PROPORTION = 0.15
MASK_CONSECUTIVE = 1
LENGTH_ROUTINE = 'CONSTANT'

def down_sample_frames(spec, dr):
    left_over = spec.shape[1] % dr
    if left_over != 0: spec = spec[:, :-left_over, :]
    spec_stacked = spec.view(spec.shape[0], spec.shape[1]//dr, spec.shape[2]*dr)
    return spec_stacked


def position_encoding(seq_len, hidden_size, batch_size=None, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        sinusoid_table[padding_idx:] = 0. # zero vector for padding dimension

    if batch_size is not None:
        batch_sinusoid_table = np.repeat(sinusoid_table[np.newaxis,...], batch_size, axis=0)
        return batch_sinusoid_table # (batch_size, seq_len, hidden_size)
    else:
        return sinusoid_table  # (seq_len, hidden_size)


def process_train_MAM_data(spec, myFiles, config=None, adaptation=-1):
    """Process training data for the masked acoustic model"""

    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE
    mask_proportion = config['mask_proportion'] if config is not None else MASK_PROPORTION
    mask_consecutive = config['mask_consecutive'] if config is not None else MASK_CONSECUTIVE
    try:
        length_routine = config['length_routine'] if config is not None else LENGTH_ROUTINE
    except:
        length_routine = LENGTH_ROUTINE
    with torch.no_grad():
        if len(spec) == 2: # if self.duo_feature: dataloader will output `source_spec` and `target_spec`
            source_spec = spec[0]
            target_spec = spec[1]
        elif len(spec) == 1:
            source_spec = spec[0]
            target_spec = copy.deepcopy(spec[0])
        else:
            raise NotImplementedError('Input spec sould be either (spec,) or (target_spec, source_spec), where `spec` has shape BxTxD.')

        # Down sample
        spec_masked = down_sample_frames(source_spec, dr) # (batch_size, seq_len, mel_dim * dr)
        spec_stacked = down_sample_frames(target_spec, dr) # (batch_size, seq_len, mel_dim * dr)
        assert(spec_masked.shape[1] == spec_stacked.shape[1]), 'Input and output spectrogram should have the same shape'

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]
        #print(spec_len)
        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, hidden_size, batch_size) # (batch_size, seq_len, hidden_size)
        mask_label = np.zeros_like(spec_stacked)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)
        
        #CONSTANT LENGTH MASKING (ORIGINAL)
        if length_routine == 'CONSTANT':
            for idx in range(len(spec_stacked)):
            
                # determine whether to mask / random / or do nothing to the frame
                dice = torch.rand(1).data.cpu()
                valid_index_range = int(spec_len[idx] - mask_consecutive - 1) # compute valid len for consecutive masking
                proportion = int(spec_len[idx]*324* mask_proportion // (mask_consecutive*mask_consecutive))
                #print(proportion)
                #print(valid_index_range)
                #print(spec_len[idx])
                #print(idx)
                #h = torch.randperm(valid_index_range)
                if valid_index_range<0:
                    #print("HOGOOEOEOOEE")
                    print(myFiles[idx])
                    valid_index_range = 0
                    with open('bad_files.txt','a') as myFile:
                        myFile.write(myFiles[idx])
                        myFile.write('\n')
                chosen_index2 = torch.randperm(324).data.cpu().numpy()[:proportion] 
                chosen_index1 = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion] # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                #print(chosen_index)
                # mask to zero
                #TODO: Mask the features in this way:
                #print(mask_consecutive)
                #ssssssss
                #chosen_index_feat = torch.randperm(272).data.cpu().numpy()[0]
                #chosen_index_feat2 = torch.randperm(272-chosen_index_feat).data.cpu().numpy()[0]
                #print(chosen_index_feat)
                #spec_masked[idx][:,52+chosen_index_feat:52+chosen_index_feat+chosen_index_feat2] = 0
                #print(spec_masked[idx][1,:])
                #print(spec_masked[idx][1,:].shape)
                #kkk
                if bool(dice < 0.8):
                    for k in range(len(chosen_index1)):
                        for i in range(mask_consecutive):
                        #print(spec_masked.shape)
                        #TODO: Had to fix so gradients are taken on the right frames
                            for j in range(mask_consecutive):
                                #print(chosen_index)
                                #print(spec_len[idx])
                                #print(chosen_index%324)
                                #sss
                                spec_masked[idx][min(chosen_index1[k]+i,spec_len[idx]),min(chosen_index2[k]+j,323)] = 0
                # replace to random frames
                elif bool(dice >= 0.8) and bool(dice < 0.9):
                    random_index = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion]
                    for k in range(len(chosen_index1)):
                        for i in range(mask_consecutive):
                            for j in range(mask_consecutive):
                                spec_masked[idx][min(chosen_index1[k]+i,spec_len[idx]),min(chosen_index2[k]+j,323)] = spec_masked[idx][min(random_index[k]+i,spec_len[idx]),min(chosen_index2[k]+j,323)]
                # do nothing
                else:
                    pass

                # TODO: I fixed this by adjusting for mask_consecutive
                #the gradients will be calculated on all chosen frames
                #mask_label[idx][chosen_i = 1
                #print(sum(mask_label[idx]))

                #for i in range(mask_consecutive):
                #    mask_label[idx][chosen_index+i] = 1
                #print(sum(mask_label[idx]))            
                #TODO: Add frames masking as well!
                for k in range(len(chosen_index1)):
                    for i in range(mask_consecutive):
                        for j in range(mask_consecutive):
                            mask_label[idx][min(chosen_index1[k]+i,spec_len[idx]),min(chosen_index2[k]+j,323)]=1
                # zero vectors for padding dimension
                #print(sum(mask_label[idx]))
                #sssss

        #ELSE USE GEOMETRIC SEQUENCE LENGTHS
        else:
            #print(length_routine)
            #ssssss
            for idx in range(len(spec_stacked)):
                frames_altered = 0
                #correction = 0
                my_inds = [-1000]
                #TODO: Adjust minimum/maximum length?
                lmin = 2
                lmax = 7
                #TODO use adaptation
                if adaptation != -1:
                    lmax = adaptation
                    mask_consecutive = adaptation
                #print(adaptation)
                frames_budget = (spec_len[idx]*mask_proportion)//1
                # determine whether to mask / random / or do nothing to the frame
                dice = torch.rand(1).data.cpu()

                while frames_altered < frames_budget:
                    s = np.random.randint(spec_len[idx])
                    if min([abs(s-kk) for kk in my_inds])<lmax:
                        continue
                    l = np.random.geometric(1.0/mask_consecutive)
                    #xxxx TODO: use this for the ablation!
                    #l = 7
                    #CLAMP to min (and optionally max)
                    l = lmin if l < lmin else l
                    l = lmax if l > lmax else l
                    #TODo: Remove correction!
                    
                    #Don't go out of bounds
                    l = (spec_len[idx]-s) if s+l > spec_len[idx] else l 
                    #--- here to nex ---  is new
                    #replace temp_ind with s
                    #---
                    if bool(dice < 0.8):
                        spec_masked[idx][s:s+l] = 0
                    # replace to random frames
                    elif bool(dice >= 0.8) and bool(dice < 0.9):
                        random_index = torch.randperm(spec_len[idx]).data.cpu().numpy()[:l]
                        spec_masked[idx][s:s+l] = spec_masked[idx][random_index]
                    # do nothing
                    else:
                        pass
                    mask_label[idx][s:s+l] = 1

                    
                    frames_altered += l
                    
                #diff = [my_inds[kk + 1] - my_inds[kk] for kk in range(len(my_inds)-1)]
                #if min(diff) < 50:
                #    ssssssss
                #chosen_index = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion] # draw `proportion` samples from the range (0, valid_index_range) and without replacement
                #print(chosen_index)
                # mask to zero
                #TODO: Mask the features in this way:
                chosen_index_feat = torch.randperm(272).data.cpu().numpy()[0]
                chosen_index_feat2 = torch.randperm(272-chosen_index_feat).data.cpu().numpy()[0]
                #print(chosen_index_feat)
                spec_masked[idx][:,52+chosen_index_feat:52+chosen_index_feat+chosen_index_feat2] = 0
                #print(spec_masked[idx][1,:])
                #print(spec_masked[idx][1,:].shape)
                #kkk
                mask_label[idx][:,52+chosen_index_feat:52+chosen_index_feat+chosen_index_feat2]=1
                # zero vectors for padding dimension
                #print(sum(mask_label[idx]))
                #sssss


            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0

        spec_masked = spec_masked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)
        spec_stacked = spec_stacked.to(dtype=torch.float32)

    return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked


def process_test_MAM_data(spec, config=None):
    """Process testing data for the masked acoustic model"""
    
    dr = config['downsample_rate'] if config is not None else DR
    hidden_size = config['hidden_size'] if config is not None else HIDDEN_SIZE

    with torch.no_grad():
        if len(spec) != 1:
            raise NotImplementedError('Input spec sould be a tuple of: (spec,), where `spec` has shape BxTxD.')

        # Down sample
        spec_stacked = down_sample_frames(spec[0], dr) # (batch_size, seq_len, mel_dim * dr)

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, hidden_size, batch_size) # (batch_size, seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            pos_enc[idx][spec_len[idx]:] = 0  
            attn_mask[idx][spec_len[idx]:] = 0 

        spec_stacked = spec_stacked.to(dtype=torch.float32)
        pos_enc = torch.FloatTensor(pos_enc).to(dtype=torch.float32)
        attn_mask = torch.FloatTensor(attn_mask).to(dtype=torch.float32)

    return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)



