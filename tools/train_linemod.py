import sys
import scipy.misc
from tqdm import tqdm
from datetime import date
from pathlib import Path
import gc
import GPUtil
import copy

from skimage.io import imsave
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

sys.path.append('.')
sys.path.append('..')
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean, ransac_voting_layer_v5, ransac_motion_voting
from lib.networks.model_repository import *
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType, RandomScaleCrop
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, TruncatedLineModImageDB
from lib.utils.arg_utils import args
from lib.utils.draw_utils import visualize_bounding_box, imagenet_to_uint8, visualize_mask, visualize_points, img_pts_to_pts_img
from lib.utils.base_utils import save_pickle
import json

from lib.utils.evaluation_utils import Evaluator
from lib.utils.net_utils import AverageMeter, Recorder, smooth_l1_loss, \
    load_model, save_model, adjust_learning_rate, compute_precision_recall, set_learning_rate,\
         compute_step_size, perturb_gt_input, load_pretrained_estNet, load_model_estNet, load_model_imNet, \
             save_model_estNet, save_model_imNet, load_pretrained_imNet

from lib.utils.config import cfg

from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch
import torch.nn.functional as F
import os
import time
from collections import OrderedDict
import random
import numpy as np
import time
from gerards_functions import *


with open(args.cfg_file,'r') as f:
    train_cfg=json.load(f)
train_cfg['model_name']='{}_{}'.format(args.linemod_cls,train_cfg['model_name'])
train_cfg['exp_name'] = '{}_{}'.format(train_cfg['exp_name'],train_cfg['object'])

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

loss_rec = AverageMeter()
precision_rec = AverageMeter()
recall_rec = AverageMeter()
q_loss_rec = AverageMeter()
recs=[loss_rec, precision_rec, recall_rec]
recs_names=['scalar/ver','scalar/q', 'scalar/precision','scalar/recall']

data_time = AverageMeter()
batch_time = AverageMeter()
recorder = Recorder(True,os.path.join(cfg.REC_DIR,train_cfg['model_name']),
                    os.path.join(cfg.REC_DIR,train_cfg['model_name']+'.log'))

class NetWrapper(nn.Module):
    def __init__(self,imNet, estNet):
        super(NetWrapper,self).__init__()
        self.imNet=imNet
        self.estNet=estNet
        # self.rank=rank
        self.criterionSeg=nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights, vertex_init_pert, vertex_init):      
        
        vertex_pred, x2s, x4s, x8s, xfc = self.estNet(vertex_weights * vertex_init_pert) #
        seg_pred, q_pred = self.imNet(image, x2s, x4s, x8s, xfc)

        loss_q = smooth_l1_loss(q_pred,(vertex_init-vertex), vertex_weights, reduce=False) 
        loss_vertex = smooth_l1_loss(vertex_pred, vertex_init, vertex_weights, reduce=False)
        loss = (10*loss_vertex) + loss_q
          
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, q_pred, loss, precision, recall

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

    iterations = train_cfg["train_iterations"]
    sigma = train_cfg["sigma"]
    PVNet.eval()
    for idx, data in enumerate(dataloader):
        image, mask, vertex, vertex_weights,_,hcoords = [d for d in data]
        image = image.cuda()
        mask = mask.cuda()
        vertex = vertex.cuda()
        vertex_weights = vertex_weights.cuda()        
        im = image#.half()
        data_time.update(time.time()-end)
        with torch.no_grad():
            _, vertex_init_out = PVNet(im)
            vertex_init = vertex_init_out.float()
            del vertex_init_out

            vertex_init_pert = vertex_init#perturb_gt_input(vertex_init, hcoords.cpu(), mask)

        for i in range(iterations):
            _, _,q_pred, loss, precision, recall = net(image.detach(), mask.detach(), vertex.detach(), vertex_weights.detach(), vertex_init_pert.detach(), vertex_init.detach())
            loss, precision,recall=[torch.mean(val) for val in (loss, precision, recall)]

            q_gt = vertex_init - vertex
            vals=(loss, precision, recall)

            for rec,val in zip(recs,vals): rec.update(val)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            end=time.time()
                       
            if idx % train_cfg['loss_rec_step'] == 0:
                step = epoch * size + idx        
                losses_batch=OrderedDict()
                for name,rec in zip(recs_names,recs): losses_batch['train/'+name]=rec.avg
                recorder.rec_loss_batch(losses_batch,step,epoch)
                for rec in recs: rec.reset()

                data_time.reset()
                batch_time.reset()

            vertex_init = (vertex_init - (sigma*q_gt))
        #     torch.cuda.empty_cache()
        #     gc.collect()
        # del vertex, loss, q_gt, vertex_weights, mask        
        # torch.cuda.empty_cache()
        # gc.collect()
        
    print('epoch {} training cost {} s'.format(epoch,time.time()-train_begin))
    

def val(net, PVNet, dataloader, epoch, lr, writer, val_prefix='val', use_camera_intrinsic=False, use_motion=False):
    for rec in recs: rec.reset()

    test_begin = time.time()
    eval_net=DataParallel(EvalWrapper().cuda()) if not use_motion else DataParallel(MotionEvalWrapper().cuda())
    uncertain_eval_net=DataParallel(UncertaintyEvalWrapper().cuda())
    net.eval()
        
    iterations = train_cfg["eval_iterations"] + 1

    evaluatorList = []
    losses_vertex = np.zeros(int((iterations)))# /train_cfg["skips"])+(train_cfg["skips"]-1)))
    losses_q = np.zeros(int((iterations)))# /train_cfg["skips"])+(train_cfg["skips"]-1)))
    norm_q = np.zeros(int((iterations)))# /train_cfg["skips"])+(train_cfg["skips"]-1)))
    norm_v = np.zeros(int((iterations)))# /train_cfg["skips"])+(train_cfg["skips"]-1)))
    deltas = np.zeros(int((iterations)))# /train_cfg["skips"])+(train_cfg["skips"]-1)))
    for i in range(iterations):
        # if (i % train_cfg["skips"]==0) or (i==0):
        evaluatorList.append(Evaluator())

    data_counter = 0   
    PVNet.eval()
    for idx, data in tqdm(enumerate(dataloader)):
        if use_camera_intrinsic:
            image, mask, vertex, vertex_weights, pose, corner_target, Ks = [d.cuda() for d in data]
        else:
            image, mask, vertex, vertex_weights, pose, corner_target = [d.cuda() for d in data]           

        with torch.no_grad():
            
            pose=pose.cpu().numpy()
            delta = train_cfg["delta"]
            id = 0
            for t in range(iterations): 
                if t==0:
                    seg_pred, vertex_init = PVNet(image)
                    mask_init = torch.argmax(seg_pred,1)
                    vertex_init = vertex_init.float()
                else: 
                    vertex_init_pert = vertex_init
                    _, _, q_pred, loss, precision, recall = net(image, mask, vertex, vertex_weights, vertex_init_pert.detach(), vertex_init.detach())
                    loss, precision, recall=[torch.mean(val) for val in ( loss, precision, recall)]
                    # losses_seg[t] = losses_seg[t] + loss_seg      
                    # losses_vertex[id] = losses_vertex[id] + loss_vertex
                    # losses_q[id] = losses_q[id] + loss_q
                    
                    # q_gt = vertex_init - vertex
                    norm_q[id] = norm_q[id] + ((1/torch.sum(vertex_weights).cpu().numpy()) * \
                        (np.linalg.norm((vertex_weights.cpu().numpy() * (q_pred.cpu().numpy()  - (vertex_init.cpu().numpy() - vertex.cpu().numpy())))**2)))

                    if t>0:
                        # print(delta)
                       
                        # delta = compute_step_size(delta, vertex_init.detach(), vertex_weights.detach(), q_pred.detach(), train_cfg,t)
                        deltas[id] = deltas[id] + delta
                        # if not  delta == 0:
                        vertex_init = vertex_init - (delta*q_pred)
                        # vertex_init = normalise_vector_field(vertex_init,vertex_init0)

                    # if data_counter==1000:
                    #     plot_mask_vfield(image, rgb_pth, mask_init, mask_pth, vertex_init, t)               
                if not torch.sum(vertex_weights).cpu().numpy()==0:
                    norm_v[id] = norm_v[id] + ((1/torch.sum(vertex_weights).cpu().numpy()) * \
                            (np.linalg.norm((vertex_weights.cpu().numpy() * (vertex_init.cpu().numpy()- vertex.cpu().numpy())))**2)) 

                if not train_cfg["gadi"]:
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
                            pose_preds.append(evaluatorList[id].evaluate_uncertainty(mean[bi],cov_inv[bi],pose[bi],args.linemod_cls,
                                                                        intri_type,vote_type,intri_matrix=K))
                        else:
                            pose_preds.append(evaluatorList[id].evaluate(corner_pred[bi],pose[bi],args.linemod_cls,intri_type,
                                vote_type,intri_matrix=K))
            
                if t>0:
                    vals=[loss,precision,recall]
                    for rec,val in zip(recs,vals): rec.update(val)

                if t==0:
                    corners_init = corner_pred[np.newaxis,...]
                if t==iterations-1:
                    corners_end = corner_pred[np.newaxis,...]
                    visualize_bounding_box(image,corners_init,t,corners_targets=corners_end)
                id+=1 
            # print(delta)

        data_counter+=1
    
    proj_err_list = []
    add_list = []
    p_increase_add = 0
    p_decrease_v = 0
    p_decrease_q = 0 
    first_a = []
    first_v = []    
    id = 0
    for i in range(iterations):
        proj_err,add,cm=evaluatorList[id].average_precision(False)
        norm_q[id] = norm_q[id] / len(dataloader)
        norm_v[id] = norm_v[id] / len(dataloader)           
        proj_err_list.append(proj_err)
        add_list.append(add)
        deltas[id] = deltas[id] / len(dataloader)
        id+=1

    print(train_cfg['exp_name'])
    print('delta: ',train_cfg['delta'])
    print('iterations: ',train_cfg['eval_iterations'])    
    print('X-X^ list: ',norm_v)
    print("add list: ",add_list)
    print('proj error list: ',proj_err_list)
    print("add final: ",add_list[-1])
    print("proj final: ",proj_err_list[-1])
    p_increase_add = ((add_list[-1]- add_list[0])/add_list[0])*100
    p_decrease_p = ((proj_err_list[-1] - proj_err_list[0])/proj_err_list[0]) * 100
    print('add percentage increase ', p_increase_add)
    print('Proj percentage increase: ',p_decrease_p)

#    plot_results(train_cfg, add_list, epoch, val_prefix)


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
        and val_prefix == 'val'): # or args.test_model:
        print(val_prefix)
        proj_err,add,cm=evaluatorList[-1].average_precision(False)
        losses_batch['{}/scalar/projection_error'.format(val_prefix)]=proj_err
        losses_batch['{}/scalar/add'.format(val_prefix)]=add
        losses_batch['{}/scalar/cm'.format(val_prefix)]=cm
        recorder.rec_loss_batch(losses_batch, epoch, epoch, val_prefix)
        writer.add_scalar('projection error', proj_err, epoch)
        writer.add_scalar('add',add,epoch)
        # writer.add_scalar('vertex loss',loss_vertex, epoch)
        # writer.add_scalar('q loss',loss_q, epoch)
        # writer.add_scalar('seg loss', loss_seg, epoch)
        writer.add_scalar('learning rate', lr, epoch)

    for rec in recs: rec.reset()

    print('epoch {} {} cost {} s'.format(epoch,val_prefix,time.time()-test_begin))

    return add_list, first_a, first_v, add_list[-1], norm_v[-1], proj_err_list[-1], p_increase_add, p_decrease_v, p_decrease_q
    
def train_net():

    tf_dir = './runs/' + train_cfg['exp_name']
    writer = SummaryWriter(log_dir=tf_dir)
    # Path("/home/gerard/myPvnet/pvnet/{}".format(train_cfg["exp_name"])).mkdir(parents=True, exist_ok=True)
    model_dir=os.path.join(cfg.MODEL_DIR,train_cfg['model_name'])
    # model_dir = '/home/gerard/wacvModels/{}'.format(train_cfg['object'])
    folders = model_dir.split(train_cfg['model_name'])
    # resultsFileName = folders[0] + "{}_eval.txt".format(train_cfg['model_name'])
    # resultsFile = open(resultsFileName,"a+")

    imNet=ImageUNet(ver_dim=(vote_num*2), seg_dim=2)
    estNet = EstimateUNet(ver_dim=(vote_num*2), seg_dim=2)
    net=NetWrapper(imNet,estNet)
    net=DataParallel(net).cuda()

    PVModelDir='/home/gerard/baseline_models/{}_baseline/199.pth'.format(train_cfg['object'])
    # PVModelDir='/home/gerard/xin_models/SEP_orgArc_{}_linemod_train/99.pth'.format(train_cfg['object'])
    PVNet=PVnet(ver_dim=vote_num*2, seg_dim=2)
    PVNet.load_state_dict(torch.load(PVModelDir)['net'])
    # PVNet = PVNet.half()
    PVNet = DataParallel(PVNet).cuda()
    
    randomCropping = RandomScaleCrop()
    
    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    motion_model=train_cfg['motion_model']
    print('motion state {}'.format(motion_model))

    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    image_db = LineModImageDB(args.linemod_cls,has_render_set=False,
                                has_fuse_set=False)
    test_db = image_db.test_real_set+image_db.val_real_set
    test_set = LineModDatasetRealAug(test_db, randomCropping, cfg.LINEMOD, vote_type, augment=False, use_motion=motion_model)
    test_sampler = SequentialSampler(test_set)
    test_batch_sampler = ImageSizeBatchSampler(test_sampler, train_cfg['test_batch_size'], False)
    test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=16)
    if args.linemod_cls in cfg.occ_linemod_cls_names:
        occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
        occ_test_db = occ_image_db.test_real_set
        occ_test_set = LineModDatasetRealAug(occ_test_db, randomCropping, cfg.OCCLUSION_LINEMOD, vote_type,
                                                augment=False, use_motion=motion_model)
        occ_test_sampler = SequentialSampler(occ_test_set)
        occ_test_batch_sampler = ImageSizeBatchSampler(occ_test_sampler, train_cfg['test_batch_size'], False)
        occ_test_loader = DataLoader(occ_test_set, batch_sampler=occ_test_batch_sampler, num_workers=16)
        
    if args.test_model:
        begin_epoch=load_model(net.module.imNet, net.module.estNet, optimizer, model_dir, args.load_epoch)
        
        if args.normal:
            print('testing normal linemod ...') 

            
            val_db= image_db.test_real_set+image_db.val_real_set
            val_set = LineModDatasetRealAug(val_db, randomCropping, cfg.LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
            val_sampler = SequentialSampler(val_set)
            val_batch_sampler = ImageSizeBatchSampler(val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
            val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, num_workers=16)
            # print('test with val loader')
            # _,_,_,_,_,_,_,_,_ = val(net, PVNet, val_loader, begin_epoch, lr, writer, use_motion=motion_model)           
            
            prefix='test' if args.use_test_set else 'val'
            print('test with test_loader')
            _,_,_,_,_,_,_,_,_ = val(net, PVNet, test_loader, begin_epoch, lr, writer, use_motion=motion_model)

        if args.occluded and args.linemod_cls in cfg.occ_linemod_cls_names:
            print('testing occluded linemod ...')
            
            prefix='occ_test' if args.use_test_set else 'occ_val'
            _,_,_,_,_,_,_,_,_ = val(net, PVNet, occ_test_loader, begin_epoch, lr, writer, prefix, use_motion=motion_model)

        if args.truncated:
            print('testing truncated linemod ...')
            trun_image_db = TruncatedLineModImageDB(args.linemod_cls)
            print(len(trun_image_db.set))
            trun_image_set = LineModDatasetRealAug(trun_image_db.set, cfg.LINEMOD, vote_type, randomCropping, augment=False,
                                                   use_intrinsic=True, use_motion=motion_model)
            trun_test_sampler = SequentialSampler(trun_image_set)
            trun_test_batch_sampler = ImageSizeBatchSampler(trun_test_sampler, train_cfg['test_batch_size'], False)
            trun_test_loader = DataLoader(trun_image_set, batch_sampler=trun_test_batch_sampler, num_workers=16)
            prefix='trun_test'
            _,_,_,_,_,_,_,_,_ = val(net, PVNet, trun_test_loader, begin_epoch, lr, writer, prefix, True, use_motion=motion_model)

    else:
        begin_epoch=0
        if train_cfg['resume']:
            print(model_dir)
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

        train_set = LineModDatasetRealAug(train_db, randomCropping, cfg.LINEMOD, vote_type, augment=True, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        train_sampler = RandomSampler(train_set)
        train_batch_sampler = ImageSizeBatchSampler(train_sampler, train_cfg['train_batch_size'], False, cfg=train_cfg['aug_cfg'])
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=16)

        val_db= image_db.test_real_set+image_db.val_real_set
        val_set = LineModDatasetRealAug(val_db, randomCropping, cfg.LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        val_sampler = SequentialSampler(val_set)
        val_batch_sampler = ImageSizeBatchSampler(val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
        val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, num_workers=16)

        if args.linemod_cls in cfg.occ_linemod_cls_names:
            occ_image_db=OcclusionLineModImageDB(args.linemod_cls)
            occ_val_db=occ_image_db.test_real_set[:len(occ_image_db.test_real_set)//2]
            occ_val_set = LineModDatasetRealAug(occ_val_db, randomCropping, cfg.OCCLUSION_LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
            occ_val_sampler = SequentialSampler(occ_val_set)
            occ_val_batch_sampler = ImageSizeBatchSampler(occ_val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
            occ_val_loader = DataLoader(occ_val_set, batch_sampler=occ_val_batch_sampler, num_workers=16)

        add_list_list = []
        p_inc_list = []
        p_dec_v_list = []
        p_dec_q_list = []
        final_a_list = []
        final_v_list = []
        final_p_list = []
        first_a_list = []
        first_v_list = []
        epoch_count = 0
        for epoch in range(begin_epoch, train_cfg['epoch_num']):
            adjust_learning_rate(optimizer,epoch,train_cfg['lr_decay_rate'],train_cfg['lr_decay_epoch'])
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            # train(net, PVNet, optimizer, train_loader, epoch)

            if not train_cfg["gadi"]:

                add_list, first_a, first_v, final_a, final_v, final_p, p_inc_add, p_dec_v, p_dec_q = val(net, PVNet, test_loader, epoch, lr, writer, use_motion=motion_model)
                if (train_cfg['eval_epoch']
                    and epoch%train_cfg['eval_inter']==0
                    and epoch>=train_cfg['eval_epoch_begin']) or args.test_model: 
                    if epoch >=30:
                        add_list_list.append(add_list)
                        first_a_list.append(first_a)
                        first_v_list.append(first_v)
                        p_inc_list.append(p_inc_add)
                        p_dec_v_list.append(p_dec_v)
                        p_dec_q_list.append(p_dec_q)
                        final_a_list.append(final_a)
                        final_v_list.append(final_v)
                        final_p_list.append(final_p)
            
            fileLine = "Date/time: {}. Epoch: {}. ADD: {}. 2D Proj: {} ".format(date.today(),epoch,final_a,final_p)
            if args.linemod_cls in cfg.occ_linemod_cls_names:
                print('occ')
                _,_,_, final_a_occ, _,_,_,_,_ = val(net, PVNet, occ_test_loader, epoch, lr, writer, 'occ_val',use_motion=motion_model)
                fileLine = fileLine + "Occ ADD: {}.".format(final_a_occ)
            fileLine = fileLine + '\n'
            # resultsFile.write(fileLine)
            
            print('current model: ',os.path.join(model_dir, '{}.pth'.format(epoch)))
            print('---------------------------------------------')
            # save_model(net.module.imNet, net.module.estNet, optimizer, epoch, model_dir)
            
            epoch_count+=1
        if ~train_cfg['gadi']:
            print(train_cfg['exp_name'])
            print('PVNet ADD. mean: {} +/- {}, max: {}'.format(np.mean(first_a_list),np.std(first_a_list),np.max(first_a_list)))
            # print('PVNet X-X^. mean: {} +/- {}, max: {}'.format(np.mean(first_v_list),np.std(first_v_list),np.max(first_v_list)))
            print('ADD. mean: {} +/- {}, max: {}: '.format(np.mean(final_a_list),np.std(final_a_list),np.max(final_a_list)))
            print('ADD perc increase. mean: {} +/- {}, max: {}'.format(np.mean(p_inc_list),np.std(p_inc_list),np.max(p_inc_list)))
            print('X-X^ perc decrease. mean: {} +/- {}, max: {}'.format(np.mean(p_dec_v_list),np.std(p_dec_v_list),np.max(p_dec_v_list)))
            # print('q-q^ perc decrease. mean: {} +/- {}, max: {}'.format(np.mean(p_dec_q_list),np.std(p_dec_q_list),np.max(p_dec_q_list))) 
            # resultsFile.close()

if __name__ == "__main__":
    train_net()
