import scipy
def mask_and_save(image,mask,idx):
# apply mask and save image to file
# image = torch tensor (1,3,r,c)
# mask = torch tensor (1,1,r,c)
    im = image.squeeze(0).cpu().float()
    masked_im = mask.squeeze(0).cpu().float()
    masked_image = im * masked_im
    masked_image = masked_image.numpy()
    masked_image = masked_image.transpose(1,2,0)
    scipy.misc.imsave('/home/gerard/glue_images/{}.jpg'.format(idx),masked_image)

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from datetime import date
def plot_results(train_cfg,add_list,norm_v, pose_diffs, epoch,val_prefix):
    add_list = [0.495, 0.645]
    distance = np.array(list(range(len(add_list)))) * 1.2#train_cfg["delta"])
    add_list_perc = [i * 100 for i in add_list]
    fig1 = plt.figure(figsize=[12,12])
    plt.plot(distance,add_list_perc)
    plt.ylabel('ADD(S)')
    plt.xlabel(r"$\rho$")
    plt.grid()    
    
    # fig2 = plt.figure(figsize=[30,15])
    # ax1 = plt.subplot(121)
    # ax1.plot(distance,add_list_perc)
    # ax1.set_xlabel(r"$\rho$")
    # ax1.set_ylabel('ADD(S)')
    # ax1.grid()
    # ax2 = plt.subplot(122)
    # ax2.plot(distance[1:],norm_v[1:]/norm_v[1])
    # ax2.set_ylabel(r'$||X-\hat{X}||^2_2$')
    # ax2.set_xlabel(r"$\rho$")
    # ax2.grid()   
    # ax3 = plt.subplot(133)
    # ax3.plot(distance,pose_diffs/pose_diffs[0])
    # ax3.set_ylabel(r'$||I_4 - P\hat{P}^{-1}||^2_F$') 
    # ax3.set_xlabel(r"$\rho$")
    # ax3.grid()
    
    # ax1.axvline(x=(train_cfg["train_iterations"]) * train_cfg["sigma"],color='gray',linestyle='--')
 
  
    # plt.suptitle(r"""Date: {}, Epoch: {} 
    #     $T_T$ = {}, $\sigma$ = {}: $\rho_T$ = {}. 
    #     $T_E$ = {}, $\delta$ = {}: $\rho_E$ = {}.""".format( \
    #     date.today(),epoch,(train_cfg["train_iterations"]),train_cfg["sigma"],(train_cfg["train_iterations"])*train_cfg["sigma"],\
    #     (train_cfg["eval_iterations"]),train_cfg["delta"],(train_cfg["eval_iterations"])*train_cfg["delta"]))
    # imNameEPS = '{}_{}_{}_{}_{}.eps'.format(date.today(),epoch-1,train_cfg["eval_iterations"],val_prefix,train_cfg['object'])
    # imNamePNG = '{}_{}_{}_{}_{}.png'.format(date.today(),epoch-1,train_cfg["eval_iterations"],val_prefix,train_cfg['object'])
    # fig1.savefig(imNameEPS,format='eps')
    # fig1.savefig(imNamePNG)

    imNameEPS = '{}_{}_{}_{}_{}_both.eps'.format(date.today(),epoch-1,train_cfg["eval_iterations"],val_prefix,train_cfg['object'])
    imNamePNG = '{}_{}_{}_{}_{}_both.png'.format(date.today(),epoch-1,train_cfg["eval_iterations"],val_prefix,train_cfg['object'])
    fig1.savefig(imNameEPS,format='eps')
    fig1.savefig(imNamePNG)


def compute_step_size(alpha, vertex_pred, vertex_weights, q_pred, train_cfg,t):
    c1 = 1
    c2 = 0.9
    maxItr = 50
    eta = 1-(1/maxItr)

    itr = 1
    while True:    
        a = (vertex_weights * q_pred).squeeze(0)
        b = torch.transpose((vertex_weights * q_pred),2,3).squeeze(0)
        lhs_1 = 0.5*torch.norm(vertex_weights * (vertex_pred - alpha*q_pred))**2
        rhs_1 = 0.5*torch.norm(vertex_weights * (vertex_pred))**2 + c1*(torch.norm(torch.bmm(b,a))**2)

        d = torch.transpose((vertex_weights * (vertex_pred - alpha*q_pred)),2,3).squeeze(0)
        e = (vertex_weights * q_pred).squeeze(0)
        lhs_2 = torch.norm(torch.bmm(d,e))
        rhs_2 = c2*(torch.norm(torch.bmm(b,a))**2)
        # print(lhs_1)
        # print(c1*(torch.norm(torch.bmm(b,a))**2))
        # print(rhs_1)
        # print(lhs_2)
        # print(rhs_2)
        # print('')

        if (lhs_1 <= rhs_1) and (lhs_2 >= rhs_2):
            break
        if itr > maxItr:
            alpha = 0
            break

        alpha=eta*alpha
        itr+=1
    
    # plt.figure(figsize=[12,6])
    # ax1 = plt.subplot(121)
    # ax1.plot(alphas,objectives,'b-',alphas,armijo,'r--')
    # ax1.set_ylabel('objective')
    # ax1.set_xlabel(r'$\alpha$')
    # ax2 = plt.subplot(122)
    # ax2.plot(alphas,d_objectives,'b-',alphas,d_armijo,'r--')
    # ax2.set_xlabel(r'$\alpha$')
    # ax2.set_ylabel('curvature')
    # plt.savefig('{}/{}_{}_{}.png'.format(train_cfg["exp_name"],date.today(),train_cfg["delta"],t))
    # print('--alpha: {}--itr: {}'.format(alpha,itr))          

    return alpha

import torch
def perturb_gt_input(vertex_init, hcoords, mask):
    vertex_init_zeros = torch.zeros(vertex_init.shape)
    vertex_init_pert = vertex_init_zeros
    for b in range(hcoords.shape[0]):  # for image in batch
        perturbation = torch.from_numpy((0.02 * np.random.random([9,2])) - 0.01)
        hcoords[b,:,0:2] = hcoords[b,:,0:2].double() + perturbation.double()
        v = ld.compute_vertex_hcoords(mask[b,:,:].cpu().numpy(),hcoords[b,:,:].cpu().numpy())
        v = torch.from_numpy(v)
        v = v.permute(2,0,1)
        vertex_init_pert[b,:,:,:] = v
    vertex_init_pert = vertex_init_pert.cuda()
    return vertex_init_pert

def perturb_vertex_input(vertex_init, mask):
    perturbation = np.random.random(vertex_init.shape) - 0.5
    
    perturbed = vertex_init.cpu().numpy() + perturbation 
    mask_perturbed = np.multiply(mask.cpu().numpy() , perturbed)
    return torch.from_numpy(mask_perturbed).cuda().float()
    
    


# def normalise_vector_field(vertex_init,vertex,vertex_weights):
    # normalise batch of vector fields to [-1,1] so that all values maintain original sign
    # batch_size = vertex_init.shape[0]
    # vfields = vertex_init.shape[1]
    # norm = []
    # for im in range(batch_size):
    #     v = 0
    #     current_norm = 0
    #     for vfield in range(int(vfields/2)): 
            # b = torch.max(batch0[im,vfield,:,:])
            # a = torch.min(batch0[im,vfield,:,:])
            # if torch.abs(torch.min(batch[im,vfield,:,:])) > torch.max(batch[im,vfield,:,:]):
            #     max_range_value = torch.abs(torch.min(batch[im,vfield,:,:]))
            #     min_range_value = torch.min(batch[im,vfield,:,:])
            # else:
            #     max_range_value = torch.max(batch[im,vfield,:,:])
            #     min_range_value = -torch.max(batch[im,vfield,:,:]) 
            
            # batch[im,vfield,:,:] = (b-a)*((batch[im,vfield,:,:] - min_range_value) / (max_range_value - min_range_value)) + a
            # batch[im,vfield,:,:] = torch.from_numpy(batch[im,vfield,:,:].cpu().numpy() / np.linalg.norm(batch[im,vfield].cpu().numpy()))
    #         current_norm = current_norm + ((1/torch.sum(vertex_weights[im,:,:,:]).cpu().numpy()) * \
    #                     (np.linalg.norm((vertex_weights[im,:,:,:].cpu().numpy() * (vertex_init[im,v:v+2,:,:].cpu().numpy()- vertex[im,v:v+2,:,:].cpu().numpy())))**2))
    #         v+=2
    #     norm.append(current_norm)

    # norm = np.array(norm)
    # norm = torch.from_numpy(norm).float().cuda()

    # norm = torch.mean(vertex_weights * torch.norm(vertex_init - vertex))
    # return norm
