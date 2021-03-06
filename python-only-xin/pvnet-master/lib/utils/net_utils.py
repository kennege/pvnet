import torch
import gc
from torch import nn
from easydict import EasyDict
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import lib.datasets.linemod_dataset as ld



class History:

    def load_dict(self, *args):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def update(self, *args):
        raise NotImplementedError()


class LossHistory(History):
    def __init__(self):
        self.losses = {'train': [], 'dev': []}
        self.accs = {'train': [], 'dev': []}
        self.bounding_accs = {'train': [], 'dev': []}
        self.shrink = {'train': [], 'dev': []}

    def load_dict(self, other):
        self.losses = other.losses
        self.accs = other.accs
        self.bounding_accs = other.bounding_accs
        self.shrink = other.shrink

    def plot(self):
        train_loss, train_acc, dev_loss, dev_acc, train_bound_acc, dev_bound_acc = \
            self.losses['train'], self.accs['train'], self.losses['dev'], self.accs['dev'], self.bounding_accs['train'], \
            self.bounding_accs['dev']
        train_shrink=self.shrink['train']
        dev_shrink=self.shrink['dev']
        epochs = len(train_loss)
        plt.plot(range(1, 1 + epochs), train_loss, label='train_loss')
        plt.plot(range(1, 1 + epochs), dev_loss, label='dev_loss')
        plt.plot(range(1, 1 + epochs), train_acc, label='train_acc')
        plt.plot(range(1, 1 + epochs), dev_acc, label='dev_acc')
        plt.plot(range(1, 1 + epochs), train_bound_acc, label='train_bound_acc')
        plt.plot(range(1, 1 + epochs), dev_bound_acc, label='dev_bound_acc')
        plt.plot(range(1, 1 + epochs), train_shrink, label='train_shrink')
        plt.plot(range(1, 1 + epochs), dev_shrink, label='dev_shrink')
        plt.legend()
        plt.show()

def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    '''
    :param vertex_pred:     [b,vn*2,h,w]
    :param vertex_targets:  [b,vn*2,h,w]
    :param vertex_weights:  [b,1,h,w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    '''
    b,ver_dim,_,_=vertex_pred.shape
    sigma_2 = sigma ** 2
    # vertex_diff = vertex_pred - vertex_targets
    # diff = vertex_weights * (vertex_pred - vertex_targets)
    # abs_diff = torch.abs(vertex_weights * (vertex_pred - vertex_targets))

    smoothL1_sign = (torch.abs(vertex_weights.detach() * (vertex_pred.detach() - vertex_targets.detach())) < 1. / sigma_2).detach().float()
    in_loss = torch.pow((vertex_weights.detach() * (vertex_pred - vertex_targets.detach())), 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (torch.abs(vertex_weights.detach() * (vertex_pred - vertex_targets.detach())) - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss=torch.sum(in_loss.view(b,-1),1) / (ver_dim * torch.sum(vertex_weights.view(b,-1),1) + 1e-3)

    if reduce:
       torch.mean(in_loss)

    return in_loss


def conv(num_input, num_output, kernel_size, stride, padding, relu=True):
    if relu is False:
        return nn.Conv2d(num_input, num_output, kernel_size, stride, padding)
    else:
        return nn.Sequential(
            nn.Conv2d(num_input, num_output, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )
        
def load_model(imNet, estNet, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch==-1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    # print(os.path.join(model_dir, '{}.pth'.format(pth)))
    imNet.load_state_dict(pretrained_model['imNet'])
    estNet.load_state_dict(pretrained_model['estNet'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load model {} epoch {}'.format(model_dir,pretrained_model['epoch']))
    return pretrained_model['epoch'] + 1

def load_model_estNet(estNet, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch==-1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    estNet.load_state_dict(pretrained_model['estNet'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load model {} epoch {}'.format(model_dir,pretrained_model['epoch']))
    return pretrained_model['epoch'] + 1

def load_model_imNet(imNet, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch==-1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    imNet.load_state_dict(pretrained_model['imNet'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load model {} epoch {}'.format(model_dir,pretrained_model['epoch']))
    return pretrained_model['epoch'] + 1

def load_pretrained_estNet(estNet, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        print('model not found')
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch==-1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    estNet.load_state_dict(pretrained_model['estNet'])
    print('load model {} epoch {}'.format(model_dir,pretrained_model['epoch']))
    return estNet

def load_pretrained_imNet(imNet, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        print('model not found')
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch==-1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    imNet.load_state_dict(pretrained_model['imNet'])
    print('load model {} epoch {}'.format(model_dir,pretrained_model['epoch']))
    return imNet

def load_net(net, model_dir):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0

    pth = max(pths)
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    return pretrained_model['epoch'] + 1


def save_model_estNet(estNet, optim, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'estNet': estNet.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

def save_model_imNet(imNet, optim, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'imNet': imNet.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))    

def save_model(imNet, estNet, optim, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'imNet': imNet.state_dict(),
        'estNet': estNet.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

class AverageMeter(EasyDict):
    """Computes and stores the average and current value"""
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    colors = [[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    def __init__(self, rec=True, rec_dir=None, dump_fn=None):
        from matplotlib import cm
        if rec:
            self.writer = SummaryWriter(log_dir=rec_dir)
            self.cmap = cm.get_cmap()
        else:
            self.writer = None

        self.dump_fn=dump_fn

    def rec_loss(self, loss, step, name='data/loss'):
        msg='{} {} {}'.format(name, step, loss)
        print(msg)
        if self.dump_fn is not None:
            with open(self.dump_fn,'a') as f:
                f.write(msg+'\n')

        if self.writer is None:
            return

        self.writer.add_scalar(name, loss, step)

    def rec_loss_batch(self, losses_batch, step, epoch, prefix='train'):
        msg='{} epoch {} step {}'.format(prefix, epoch, step)
        for k,v in losses_batch.items():
            msg+=' {} {:.8f} '.format(k.split('/')[-1],v)

        print(msg)
        if self.dump_fn is not None:
            with open(self.dump_fn,'a') as f:
                f.write(msg+'\n')

        if self.writer is None:
            return

        for k,v in losses_batch.items():
            self.writer.add_scalar(k, v, step)

    def rec_segmentation(self, seg, num_classes, nrow, step, name='seg'):
        if self.writer is None:
            return

        seg = torch.argmax(seg, dim=1).long()
        r = seg.clone()
        g = seg.clone()
        b = seg.clone()
        for l in range(num_classes):
            inds = (seg == l)
            r[inds] = self.colors[l][0]
            g[inds] = self.colors[l][1]
            b[inds] = self.colors[l][2]
        seg = torch.stack([r, g, b], dim=1)

        seg = vutils.make_grid(seg, nrow)
        self.writer.add_image(name, seg, step)

    def rec_vertex(self, vertex, mask, nrow, step, name='vertex'):
        if self.writer is None:
            return

        vertex = (vertex[:, :2, ...] * mask + 1) / 2
        height, width = vertex.shape[2:]
        vertex = vertex.view(-1, height, width)
        vertex = self.cmap(vertex.detach().cpu().numpy())[..., :3]
        vertex = vutils.make_grid(torch.from_numpy(vertex).permute(0, 3, 1, 2), nrow)
        self.writer.add_image(name, vertex, step)

class MultiClassPrecisionRecall:
    def __init__(self,names):
        self.class_num=len(names)
        self.names=names
        self.tp=torch.zeros(self.class_num,dtype=torch.int64).cuda()
        self.fp=torch.zeros(self.class_num,dtype=torch.int64).cuda()
        self.fn=torch.zeros(self.class_num,dtype=torch.int64).cuda()

    def accumulate(self, pred, label):
        '''
        :param pred:  b,h,w
        :param label: b,h,w
        :return:
        '''
        for ci in range(self.class_num):
            self.tp[ci]+=torch.sum((pred==ci)&(label==ci))
            self.fp[ci]+=torch.sum((pred==ci)&(label!=ci))
            self.fn[ci]+=torch.sum((pred!=ci)&(label==ci))

    def compute_precision_recall(self):
        tp=self.tp.double()
        fp=self.fp.double()
        fn=self.fn.double()
        return ((tp+1)/(tp+fp+1)).cpu().numpy(), ((tp+1)/(tp+fn+1)).cpu().numpy()

    def reset(self):
        self.tp=torch.zeros(self.class_num,dtype=torch.int64).cuda()
        self.fp=torch.zeros(self.class_num,dtype=torch.int64).cuda()
        self.fn=torch.zeros(self.class_num,dtype=torch.int64).cuda()


def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
    
    if ((epoch+1) % lr_decay_epoch)!=0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)
    print('changing learning rate {:5f} to {:.5f}'.format(lr_before,max(param_group['lr'], min_lr)))

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        lr_before = param_group['lr']
        param_group['lr'] = lr
    print('reset learning rate {:5f} to {:.5f}'.format(lr_before,lr))


def acc_img(img,pre_img_list,an,hcount,wcount,hinter,winter,height,width,model='avg'):
    if len(pre_img_list)>0:
        cur_img=torch.cat([pre_img_list.pop(),img],0)
    else:
        cur_img=img

    image_num, left_num = cur_img.shape[0] // an, cur_img.shape[0] % an

    if left_num>0: pre_img_list.append(cur_img[image_num*an:])
    if image_num==0: return None

    if model=='avg':
        h,w=cur_img.shape[2],cur_img.shape[3]
        result_img=torch.zeros([image_num, img.shape[1], height, width],dtype=img.dtype,device=img.device)
        result_wgt=torch.zeros([image_num, img.shape[1], height, width],dtype=img.dtype,device=img.device)
        cur_wgt=torch.ones_like(cur_img)
        for ii in range(image_num):
            for ai in range(an):
                hi=ai//wcount
                wi=ai%wcount
                hbeg=hi*hinter
                wbeg=wi*winter
                result_img[ii,:,hbeg:hbeg+h,wbeg:wbeg+w]+=cur_img[ii*an+ai,:,:,:]
                result_wgt[ii,:,hbeg:hbeg+h,wbeg:wbeg+w]+=cur_wgt[ii*an+ai,:,:,:]
        result_img=result_img/result_wgt
    else: # model=='assign'
        h,w=cur_img.shape[2],cur_img.shape[3]
        result_img=torch.zeros([image_num, img.shape[1], height, width],dtype=img.dtype,device=img.device)
        for ii in range(image_num):
            for ai in range(an):
                hi=ai//wcount
                wi=ai%wcount
                hbeg=hi*hinter
                wbeg=wi*winter
                result_img[ii,:,hbeg:hbeg+h,wbeg:wbeg+w]=cur_img[ii*an+ai,:,:,:]
        result_img=result_img

    return result_img

def compute_precision_recall(scores,target,reduce=False):
    b=scores.shape[0]
    preds=torch.argmax(scores,1)
    preds=preds.float()
    target=target.float()
    # print(preds.shape)
    tp=preds*target
    fp=preds*(1-target)
    fn=(1-preds)*target

    tp=torch.sum(tp.view(b,-1),1)
    fn=torch.sum(fn.view(b,-1),1)
    fp=torch.sum(fp.view(b,-1),1)

    precision=(tp+1)/(tp+fp+1)
    recall=(tp+1)/(tp+fn+1)

    if reduce:
        precision,recall=torch.mean(precision), torch.mean(recall)
    return precision, recall

def compute_precision_multi_class(scores,target,reduce=False):
    b,_,h,w=scores.shape
    preds=torch.argmax(scores,1)
    correct=preds==target
    precision=torch.sum(correct.view(b,-1),1).float()/(h*w)

    if reduce:
        precision=torch.mean(precision)

    return precision

def plot_mask_vfield(image, rgb_pth, mask_init, mask_pth, vertex_init,t):
    fnameM = '/home/gerard/masks/{}'.format(mask_pth[0]) 
    fnameM = fnameM.replace(".jpg","_{}.jpg".format(t))
    fnameM = fnameM.replace(".png","_{}.png".format(t))
    maskOut = mask_init.squeeze(0).cpu().numpy()
    imOut = image[0,:,:,:].permute(1,2,0).cpu().numpy()
    imOut = ((imOut - imOut.min()) * (1/(imOut.max() - imOut.min()) * 255))
    plt.figure()
    plt.imshow(imOut.astype('uint8'))
    plt.imshow(maskOut,cmap='jet',alpha=0.5)
    print(fnameM)
    plt.savefig(fnameM)                   
    
    fnameR = '/home/gerard/vfields/{}'.format(rgb_pth[0])
    plt.figure()
    plt.imshow(imOut.astype('uint8'))
    step = 16
    X,Y = np.meshgrid(np.arange(0,image.shape[3],step), np.arange(0,image.shape[2],step))
    U = vertex_init[0,3,:,:].cpu().numpy()
    U = ((U - U.min()) * (1/(U.max() - U.min())))
    U = U[::step,::step]
    V = vertex_init[0,2,:,:].cpu().numpy()
    V = ((V - V.min()) * (1/(V.max() - V.min())))
    V = V[::step,::step]
    
    plt.quiver(X,Y,U,V, scale=3, scale_units='inches', color='r')
    plt.savefig(fnameR)
    
    fnameR = fnameR.replace(".jpg","_{}.jpg".format(t))
    fnameR = fnameR.replace(".png","_{}.png".format(t))
    print(fnameR)
    plt.savefig(fnameR)  


def compute_step_size(alpha, vertex, vertex_pred, vertex_weights, q_pred, train_cfg,t):
    c1 = 0.0001
    c2 = 0.9
    maxItr = 100
    eta = 1-(1/maxItr)
    objectives = []
    armijo = []
    alphas = []
    d_objectives = []
    d_armijo = []
    itr = 1
    while True:
        lhs_1 = 0.5*torch.norm(vertex_weights * (vertex - (vertex_pred + alpha*q_pred)))**2
        rhs_1 = (0.5*torch.norm(vertex_weights * (vertex-vertex_pred))**2) - (c1*alpha*torch.norm(vertex_weights * (q_pred))**2)
        lhs_2 = torch.norm(vertex_weights * (vertex - (vertex_pred + alpha*q_pred)),1)
        rhs_2 = c2*torch.norm(vertex_weights * (vertex - vertex_pred),1)
        # print(lhs_1)
        # print(rhs_1)
        # print(lhs_2)
        # print(rhs_2)
        # print('')

        if (lhs_1 <= rhs_1) and (lhs_2 >= rhs_2):
            break
        if itr > maxItr:
            alpha = 0.001
            break
        objectives.append(lhs_1)
        armijo.append(rhs_1)
        d_objectives.append(lhs_2)
        d_armijo.append(rhs_2)
        alphas.append(alpha)
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
    print('------------------------------------alpha: {}--itr: {}'.format(alpha,itr))          

    return alpha


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