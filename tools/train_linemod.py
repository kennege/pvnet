import sys
import scipy.misc
from tqdm import tqdm

from skimage.io import imsave
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt

sys.path.append('.')
sys.path.append('..')
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean, ransac_voting_layer_v5, ransac_motion_voting
from lib.networks.model_repository import *
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, TruncatedLineModImageDB
from lib.utils.arg_utils import args
from lib.utils.draw_utils import visualize_bounding_box, imagenet_to_uint8, visualize_mask, visualize_points, img_pts_to_pts_img
from lib.utils.base_utils import save_pickle
import json

from lib.utils.evaluation_utils import Evaluator
from lib.utils.net_utils import AverageMeter, Recorder, smooth_l1_loss, \
    load_model, save_model, adjust_learning_rate, compute_precision_recall, set_learning_rate
from lib.utils.config import cfg

from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch
import torch.nn.functional as F
import os
import time
from collections import OrderedDict
import random
import numpy as np

import lib.datasets.linemod_dataset as ld

with open(args.cfg_file,'r') as f:
    train_cfg=json.load(f)
train_cfg['model_name']='{}_{}'.format(args.linemod_cls,train_cfg['model_name'])

if train_cfg['vote_type']=='BB8C':
    vote_type=VotingType.BB8C
    vote_num=9
elif train_cfg['vote_type']=='BB8S':
    vote_type=VotingType.BB8S
    vote_num=9
elif train_cfg['vote_type']=='Farthest':
    vote_type=VotingType.Farthest
    vote_num=9
elif train_cfg['vote_type']=='Farthest4':
    vote_type=VotingType.Farthest4
    vote_num=5
elif train_cfg['vote_type']=='Farthest12':
    vote_type=VotingType.Farthest12
    vote_num=13
elif train_cfg['vote_type']=='Farthest16':
    vote_type=VotingType.Farthest16
    vote_num=17
else:
    assert(train_cfg['vote_type']=='BB8')
    vote_type=VotingType.BB8
    vote_num=8


seg_loss_rec = AverageMeter()
ver_loss_rec = AverageMeter()
precision_rec = AverageMeter()
recall_rec = AverageMeter()
q_loss_rec = AverageMeter()
recs=[seg_loss_rec,ver_loss_rec,q_loss_rec,precision_rec,recall_rec]
recs_names=['scalar/seg','scalar/ver','scalar/q','scalar/precision','scalar/recall']

data_time = AverageMeter()
batch_time = AverageMeter()
recorder = Recorder(True,os.path.join(cfg.REC_DIR,train_cfg['model_name']),
                    os.path.join(cfg.REC_DIR,train_cfg['model_name']+'.log'))

# network_time,voting_time,load_time=[],[],[]

# poses_pr=[]
# poses_gt=[]

class NetWrapper(nn.Module):
    def __init__(self,imNet, estNet):
        super(NetWrapper,self).__init__()
        self.imNet=imNet
        self.estNet=estNet
        self.criterionSeg=nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights, vertex_init):      
        vertex_pred, x2s, x4s, x8s, xfc = self.estNet(vertex_init)
        seg_pred, q_pred = self.imNet(image, x2s, x4s, x8s, xfc)
        
        # vertex_init = est_init[:,0:18,:,:]
        # seg_in = est_pred[:,18:20,:,:]
        # vertex_pred = est_pred[:,0:18,:,:]
    
        # loss_seg = self.criterionSeg(seg_pred, abs(mask_init-mask.float()).long())
        loss_seg = self.criterionSeg(seg_pred, mask)
        # loss_seg_pred = self.criterionSeg(seg_in, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0],-1),1)
        # loss_seg_pred = torch.mean(loss_seg_pred.view(loss_seg_pred.shape[0],-1),1)

        # loss_seg = loss_seg + loss_seg_pred

        loss_vertex = smooth_l1_loss(vertex_pred, vertex_init, vertex_weights, reduce=False)
        loss_q = smooth_l1_loss(q_pred,(vertex_init-vertex), vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
    

        return seg_pred, vertex_pred, q_pred, loss_seg, loss_vertex, loss_q, precision, recall

class EvalWrapper(nn.Module):
    def forward(self, mask_pred, vertex_pred, use_argmax=True, use_uncertainty=False):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)

        if use_uncertainty:
            return ransac_voting_layer_v5(mask_pred,vertex_pred,128,inlier_thresh=0.99,max_num=100)
        else:
            return ransac_voting_layer_v3(mask_pred,vertex_pred,128,inlier_thresh=0.99,max_num=100)

class MotionEvalWrapper(nn.Module):
    def forward(self, mask_pred, vertex_pred, use_argmax=True, use_uncertainty=False):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)

        return ransac_motion_voting(mask_pred, vertex_pred)

class UncertaintyEvalWrapper(nn.Module):
    def forward(self, mask_pred, vertex_pred, use_argmax=True):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2)

        mean=ransac_voting_layer_v3(mask_pred, vertex_pred, 512, inlier_thresh=0.99)
        mean, var=estimate_voting_distribution_with_mean(mask_pred,vertex_pred,mean)
        return mean, var

def train(net, PVNet, optimizer, dataloader, epoch):
    for rec in recs: rec.reset()
    data_time.reset()
    batch_time.reset()

    train_begin=time.time()

    net.train()
    size = len(dataloader)
    end=time.time()
    for idx, data in enumerate(dataloader):
        image, mask, vertex, vertex_weights,_,_,_,_ = [d.cuda() for d in data]
        image = image.cuda()
        mask = mask.cuda()
        vertex = vertex.cuda()
        vertex_weights = vertex_weights.cuda()
        pose = pose.cuda()
        corner_targets = corner_targets.cuda()

        data_time.update(time.time()-end)
        with torch.no_grad():
            _, vertex_init = PVNet(image)
            # mask_init=torch.unsqueeze(torch.argmax(seg_init,1),1).float()     
            # est_init = torch.cat([vertex_init, seg_init],1)
            # vertex_init = vertex
       
        # vertex_init = torch.from_numpy(np.zeros((image.shape[0],18,image.shape[2], image.shape[3])))
        # for b in range(hcoords.shape[0]):  # for image in batch
        #     # for i in range(hcoords[b].shape[0]): # for coordinate in hcoords
        #         # hcoords[b][i,0:2]+=random.uniform(-0.1,0.1)*hcoords[b][i,0:2]
        #     # print(mask[b,:,:].cpu().numpy().shape)
        #     # print(hcoords[b,:,:].cpu().numpy())
        #     v = ld.compute_vertex_hcoords(mask[b,:,:].cpu().numpy(),hcoords[b,:,:].cpu().numpy())
        #     print(v)
        #     v = torch.from_numpy(v)
        #     v = v.permute(2,0,1)
        #     vertex_init[b,:,:,:] = v
        #     # print(vertex_init[b,:,:,:])
        # vertex_init = vertex_init.float()

        # vertex_init = vertex
        seg_pred, vertex_pred, q_pred, loss_seg, loss_vertex, loss_q, precision, recall = net(image, mask, vertex, vertex_weights, vertex_init.detach())
        # vertex_pred = est_pred[:,0:18,:,:]
        loss_seg, loss_vertex, loss_q, precision,recall=[torch.mean(val) for val in (loss_seg, loss_vertex, loss_q, precision, recall)]
        loss = loss_seg + (loss_vertex+loss_q) * train_cfg['vertex_loss_ratio']
        vals=(loss_seg,loss_vertex,loss_q,precision,recall)
        for rec,val in zip(recs,vals): rec.update(val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % train_cfg['loss_rec_step'] == 0:
            step = epoch * size + idx
            losses_batch=OrderedDict()
            for name,rec in zip(recs_names,recs): losses_batch['train/'+name]=rec.avg
            recorder.rec_loss_batch(losses_batch,step,epoch)
            for rec in recs: rec.reset()

            data_time.reset()
            batch_time.reset()

        if idx % train_cfg['img_rec_step'] == 0:
            batch_size = image.shape[0]
            nrow = 5 if batch_size > 5 else batch_size
            recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow, step=step, name='train/image/seg')
            recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=step, name='train/image/ver')

    print('epoch {} training cost {} s'.format(epoch,time.time()-train_begin))

def val(net, PVNet, dataloader, epoch, lr, writer, val_prefix='val', use_camera_intrinsic=False, use_motion=False):
    for rec in recs: rec.reset()

    test_begin = time.time()
    eval_net=DataParallel(EvalWrapper().cuda()) if not use_motion else DataParallel(MotionEvalWrapper().cuda())
    uncertain_eval_net=DataParallel(UncertaintyEvalWrapper().cuda())
    net.eval()
               
    if val_prefix=='val':
        iterations = 20
    else:
        iterations = 2
    evaluatorList = []
    losses_vertex = np.zeros(iterations)
    losses_seg = np.zeros(iterations)
    losses_q = np.zeros(iterations)
    for i in range(iterations):
        evaluatorList.append(Evaluator())

        
    for idx, data in tqdm(enumerate(dataloader)):
        if use_camera_intrinsic:
            image, mask, vertex, vertex_weights, pose, corner_target, Ks = [d.cuda() for d in data]
        else:
            image, mask, vertex, vertex_weights, pose, corner_target, mask_pth, rgb_pth = [d for d in data]
            image = image.cuda()
            mask = mask.cuda()
            vertex = vertex.cuda()
            vertex_weights = vertex_weights.cuda()
            pose = pose.cuda()
            corner_target = corner_target.cuda()

        with torch.no_grad():
            
            pose=pose.cpu().numpy()

            for t in range(iterations): # iterative evaluation
                if t==0:
                    seg_init, vertex_init = PVNet(image)
                    mask_init=torch.argmax(seg_init,1)
                else:              
                    seg_pred, vertex_pred, q_pred, loss_seg, loss_vertex, loss_q, precision, recall = net(image, mask, vertex, vertex_weights, vertex_init.detach())
                    loss_seg, loss_vertex, loss_q, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, loss_q, precision, recall)]
                    losses_seg[t] = losses_seg[t] + loss_seg      
                    losses_vertex[t] = losses_vertex[t] + loss_vertex
                    losses_q[t] = losses_q[t] + loss_q
                
                if (train_cfg['eval_epoch']
                    and epoch%train_cfg['eval_inter']==0
                    and epoch>=train_cfg['eval_epoch_begin']) or args.test_model:

                    if t>0:
                        delta = 1/(2*t)
                        mask_init = torch.argmax(seg_pred,1)
                        vertex_init = vertex_init - (delta*q_pred)

                    # if t==iterations-1:   
                    #     fnameM = '/home/gerard/masks/{}'.format(mask_pth[0]) 
                    #     maskOut = mask_init.squeeze(0).cpu().numpy()
                    #     imOut = image[0,:,:,:].permute(1,2,0).cpu().numpy()
                    #     plt.imshow(imOut)
                    #     plt.imshow(maskOut,cmap='jet',alpha=0.5)
                    #     plt.savefig(fnameM)
                        
                    #     fnameR = '/home/gerard/vfields/{}'.format(rgb_pth[0])
                    #     plt.figure()
                    #     plt.imshow(imOut)
    
                    #     X,Y = np.meshgrid(np.arange(0,image.shape[3]), np.arange(0,image.shape[2]))
                    #     U = vertex_init[0,3,:,:].cpu().numpy()
                    #     V = vertex_init[0,2,:,:].cpu().numpy()
                    #     print('X: ',X.shape)
                    #     print('Y: ',Y.shape)
                    #     print('U: ',U)
                    #     plt.quiver(X,Y,U,V)
                        
                    #     plt.savefig(fnameR)                    
                                                                                                            
                    if args.use_uncertainty_pnp:
                        mean,cov_inv=uncertain_eval_net(mask_init,vertex_init)
                        mean=mean.cpu().numpy()
                        cov_inv=cov_inv.cpu().numpy()
                    else: 
                        corner_pred=eval_net(mask_init,vertex_init).cpu().detach().numpy()
                        
                    b=pose.shape[0]
                    pose_preds=[]
                    for bi in range(b):
                        intri_type='use_intrinsic' if use_camera_intrinsic else 'linemod'
                        K=Ks[bi].cpu().numpy() if use_camera_intrinsic else None
                        if args.use_uncertainty_pnp:
                            pose_preds.append(evaluatorList[t].evaluate_uncertainty(mean[bi],cov_inv[bi],pose[bi],args.linemod_cls,
                                                                            intri_type,vote_type,intri_matrix=K))
                        else:
                            pose_preds.append(evaluatorList[t].evaluate(corner_pred[bi],pose[bi],args.linemod_cls,intri_type,
                                                                vote_type,intri_matrix=K))
                    
            if args.save_inter_result:
                mask_pr = torch.argmax(seg_pred, 1).cpu().detach().numpy()
                mask_gt = mask.cpu().detach().numpy()
                # assume batch size = 1
                imsave(os.path.join(args.save_inter_dir, '{}_mask_pr.png'.format(idx)), mask_pr[0])
                imsave(os.path.join(args.save_inter_dir, '{}_mask_gt.png'.format(idx)), mask_gt[0])
                imsave(os.path.join(args.save_inter_dir, '{}_rgb.png'.format(idx)),
                    imagenet_to_uint8(image.cpu().detach().numpy()[0]))
                save_pickle([pose_preds[0],pose[0]],os.path.join(args.save_inter_dir, '{}_pose.pkl'.format(idx)))
           
            if t>0:
                vals=[loss_seg,loss_vertex,loss_q,precision,recall]
                for rec,val in zip(recs,vals): rec.update(val)

    proj_err_list = []
    add_list = []
    for i in range(iterations):
        proj_err,add,cm=evaluatorList[i].average_precision(False)
        proj_err_list.append(proj_err)
        add_list.append(add)
        losses_seg[i] = losses_seg[i] / (iterations-1)
        losses_vertex[i] = losses_vertex[i] / (iterations-1)
        losses_q[i] = losses_q[i] / (iterations-1)
        print(proj_err)
        print(add)
        print(losses_seg[i])
        print(losses_vertex[i])
        print(losses_q[i])
        print('\n')

    if val_prefix == 'val':         
        fig = plt.figure(figsize=[15,8])
        ax1 = plt.subplot(231)
        ax1.plot(add_list)
        ax1.set_title('ADD')
        ax1.grid()
        ax2 = plt.subplot(232)
        ax2.plot(proj_err_list)
        ax2.set_title('2D proj')
        ax2.grid()
        ax3 = plt.subplot(233)
        ax3.plot(losses_seg[1:-1])
        ax3.set_title('loss_seg')
        ax3.grid()
        ax4 = plt.subplot(234)
        ax4.plot(losses_vertex[1:-1])
        ax4.set_title('loss_vertex')
        ax4.grid()
        ax5 = plt.subplot(235)
        ax5.plot(losses_q[1:-1])
        ax5.set_title('loss_q')
        ax5.grid()
        plt.savefig('{}.png'.format(epoch))

    with torch.no_grad():
        batch_size = image.shape[0]
        nrow = 5 if batch_size > 5 else batch_size
        recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow,
                                  step=epoch, name='{}/image/seg'.format(val_prefix))
        recorder.rec_vertex(vertex_init, vertex_weights, nrow=4, step=epoch, name='{}/image/ver'.format(val_prefix))
    
    losses_batch=OrderedDict()
    for name, rec in zip(recs_names, recs): losses_batch['{}/'.format(val_prefix) + name] = rec.avg
    if (train_cfg['eval_epoch']
        and epoch%train_cfg['eval_inter']==0
        and epoch>=train_cfg['eval_epoch_begin']
        and val_prefix == 'val') or args.test_model:
        proj_err,add,cm=evaluatorList[-1].average_precision(False)
        losses_batch['{}/scalar/projection_error'.format(val_prefix)]=proj_err
        losses_batch['{}/scalar/add'.format(val_prefix)]=add
        losses_batch['{}/scalar/cm'.format(val_prefix)]=cm
        recorder.rec_loss_batch(losses_batch, epoch, epoch, val_prefix)
        writer.add_scalar('projection error', proj_err, epoch)
        writer.add_scalar('add',add,epoch)
        writer.add_scalar('vertex loss',loss_vertex, epoch)
        writer.add_scalar('q loss',loss_q, epoch)
        writer.add_scalar('seg loss', loss_seg, epoch)
        writer.add_scalar('learning rate', lr, epoch)

    for rec in recs: rec.reset()



    print('epoch {} {} cost {} s'.format(epoch,val_prefix,time.time()-test_begin))

def train_net():
    tf_dir = './runs/' + train_cfg['exp_name']
    writer = SummaryWriter(log_dir=tf_dir)

    imNet=ImageUNet(ver_dim=(vote_num*2), seg_dim=2)
    estNet = EstimateUNet(ver_dim=(vote_num*2), seg_dim=2)
    net=NetWrapper(imNet,estNet)
    net=DataParallel(net).cuda()

    # load original pvnet to perform forward pass to get initial estimate
    PVModelDir='/home/gerard/199.pth'
    PVNet=PVnet(ver_dim=vote_num*2, seg_dim=2)
    PVNet.load_state_dict(torch.load(PVModelDir)['net'])
    PVNet=DataParallel(PVNet).cuda()
    PVNet.eval()

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    model_dir=os.path.join(cfg.MODEL_DIR,train_cfg['model_name'])
    motion_model=train_cfg['motion_model']
    print('motion state {}'.format(motion_model))

    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    if args.test_model:
        begin_epoch=load_model(net.module.imNet, net.module.estNet, optimizer, model_dir, args.load_epoch)
        
        if args.normal:
            print('testing normal linemod ...') 
            image_db = LineModImageDB(args.linemod_cls,has_render_set=False,
                                      has_fuse_set=False)
            test_db = image_db.test_real_set+image_db.val_real_set
            test_set = LineModDatasetRealAug(test_db, cfg.LINEMOD, vote_type, augment=False, use_motion=motion_model)
            test_sampler = SequentialSampler(test_set)
            test_batch_sampler = ImageSizeBatchSampler(test_sampler, train_cfg['test_batch_size'], False)
            test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=0)
            prefix='test' if args.use_test_set else 'val'
            
            val(net, PVNet, test_loader, begin_epoch, lr, writer, prefix, use_motion=motion_model)

        if args.occluded and args.linemod_cls in cfg.occ_linemod_cls_names:
            print('testing occluded linemod ...')
            occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
            occ_test_db = occ_image_db.test_real_set
            occ_test_set = LineModDatasetRealAug(occ_test_db, cfg.OCCLUSION_LINEMOD, vote_type,
                                                 augment=False, use_motion=motion_model)
            occ_test_sampler = SequentialSampler(occ_test_set)
            occ_test_batch_sampler = ImageSizeBatchSampler(occ_test_sampler, train_cfg['test_batch_size'], False)
            occ_test_loader = DataLoader(occ_test_set, batch_sampler=occ_test_batch_sampler, num_workers=0)
            prefix='occ_test' if args.use_test_set else 'occ_val'
            val(net, PVNet, occ_test_loader, begin_epoch, lr, writer, prefix, use_motion=motion_model)

        if args.truncated:
            print('testing truncated linemod ...')
            trun_image_db = TruncatedLineModImageDB(args.linemod_cls)
            print(len(trun_image_db.set))
            trun_image_set = LineModDatasetRealAug(trun_image_db.set, cfg.LINEMOD, vote_type, augment=False,
                                                   use_intrinsic=True, use_motion=motion_model)
            trun_test_sampler = SequentialSampler(trun_image_set)
            trun_test_batch_sampler = ImageSizeBatchSampler(trun_test_sampler, train_cfg['test_batch_size'], False)
            trun_test_loader = DataLoader(trun_image_set, batch_sampler=trun_test_batch_sampler, num_workers=0)
            prefix='trun_test'
            val(net, PVNet, trun_test_loader, begin_epoch, lr, writer, prefix, True, use_motion=motion_model)

    else:
        begin_epoch=0
        if train_cfg['resume']:
            begin_epoch=load_model(net.module.imNet, net.module.estNet, optimizer, model_dir)

            # reset learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_cfg['lr']
                lr = param_group['lr']

        image_db = LineModImageDB(args.linemod_cls,
                                  has_fuse_set=train_cfg['use_fuse'],
                                  has_render_set=True)

        train_db=[]
        train_db+=image_db.render_set
        if train_cfg['use_real_train']:
            train_db+=image_db.train_real_set
        if train_cfg['use_fuse']:
            train_db+=image_db.fuse_set

        train_set = LineModDatasetRealAug(train_db, cfg.LINEMOD, vote_type, augment=True, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        train_sampler = RandomSampler(train_set)
        train_batch_sampler = ImageSizeBatchSampler(train_sampler, train_cfg['train_batch_size'], False, cfg=train_cfg['aug_cfg'])
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=12)

        val_db= image_db.val_real_set#image_db.test_real_set+ # test on the full test set
        val_set = LineModDatasetRealAug(val_db, cfg.LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        val_sampler = SequentialSampler(val_set)
        val_batch_sampler = ImageSizeBatchSampler(val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
        val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, num_workers=12)

        if args.linemod_cls in cfg.occ_linemod_cls_names:
            occ_image_db=OcclusionLineModImageDB(args.linemod_cls)
            occ_val_db=occ_image_db.test_real_set[:len(occ_image_db.test_real_set)//2]
            occ_val_set = LineModDatasetRealAug(occ_val_db, cfg.OCCLUSION_LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
            occ_val_sampler = SequentialSampler(occ_val_set)
            occ_val_batch_sampler = ImageSizeBatchSampler(occ_val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
            occ_val_loader = DataLoader(occ_val_set, batch_sampler=occ_val_batch_sampler, num_workers=12)

        for epoch in range(begin_epoch, train_cfg['epoch_num']):
            adjust_learning_rate(optimizer,epoch,train_cfg['lr_decay_rate'],train_cfg['lr_decay_epoch'])
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            train(net, PVNet, optimizer, train_loader, epoch)
            val(net, PVNet, val_loader, epoch, lr, writer, use_motion=motion_model)
            if args.linemod_cls in cfg.occ_linemod_cls_names:
                val(net, PVNet, occ_val_loader, epoch, lr, writer, 'occ_val',use_motion=motion_model)

            save_model(net.module.imNet, net.module.estNet, optimizer, epoch, model_dir)

# def save_dataset(dataset,prefix=''):
#     with open('assets/{}{}.txt'.format(prefix,args.linemod_cls),'w') as f:
#         for data in dataset: f.write(data['rgb_pth']+'\n')
#
# def save_poses_dataset(prefix=''):
#     print(np.asarray(poses_pr).shape)
#     np.save('assets/{}{}_pr.npy'.format(prefix,args.linemod_cls),np.asarray(poses_pr))
#     np.save('assets/{}{}_gt.npy'.format(prefix,args.linemod_cls),np.asarray(poses_gt))

if __name__ == "__main__":
    train_net()
    # save_poses_dataset('trun_')
