# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset_unit import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet, SCMPoseRefiner
from lib.loss import Loss
from lib.loss_refiner import Loss_refine, SCMLoss
from lib.utils import setup_logger
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = 'datasets\ycb\YCB_Video_Dataset', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 128, help='batch size')
parser.add_argument('--workers', type=int, default = 16, help='number of data loading workers')
parser.add_argument('--lr', default=0.000001, help='learning rate')
parser.add_argument('--lr_rate', default=0.5, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=100, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = 'pose_model_368_0.030526180794098117.pth',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--num_obj', type=int, default=8, help='number of object classes in the dataset')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb' or opt.dataset == 'ycb_jpg':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    elif opt.dataset == 'bin':
        opt.num_points = 1500
        #opt.num_objects = 21
        opt.outf = 'trained_models/bin'
        opt.log_dir = 'experiments/logs/bin'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    # refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner = SCMPoseRefiner(num_points=opt.num_points, num_obj=opt.num_objects).float()
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=opt.lr_rate)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'ycb_jpg':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, ext='jpg')
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'ycb_jpg':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, ext='jpg')
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    scm_criterion = SCMLoss(num_points_mesh=opt.num_points_mesh, sym_list=opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        
        train_count = 0
        train_dis_avg = 0.0
        batch_count = 0
        if epoch == 1:
            opt.refine_start = True
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        total_batches = len(dataloader) * opt.repeat_epoch
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch}')

        for rep in range(opt.repeat_epoch):
                    for i, data in enumerate(dataloader, 0):
                        points, choose, img, target, model_points, idx = data
                        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                        Variable(choose).cuda(), \
                                                                        Variable(img).cuda(), \
                                                                        Variable(target).cuda(), \
                                                                        Variable(model_points).cuda(), \
                                                                        Variable(idx).cuda()
                        # Initial DenseFusion estimation
                        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                        
                        if opt.refine_start:
                            pred_r = pred_r.permute(0, 2, 1)  # Shape: (num_obj, 4, N)
                            pred_t = pred_t.permute(0, 2, 1)  # Shape: (num_obj, 3, N)
                            initial_pose = torch.cat([pred_r, pred_t], dim=1)  # Shape: (num_obj, 7, N)


                            for ite in range(0, opt.iteration):
                                # Generate random view intervention
                                interventions = {}
                                if ite % 2 == 0:
                                    # Random rotation around z-axis for view intervention
                                    angle = torch.rand(1).cuda() * 2 * np.pi
                                    c, s = torch.cos(angle), torch.sin(angle)
                                    view_transform = torch.eye(4).cuda()
                                    view_transform[0,0], view_transform[0,1] = c, -s
                                    view_transform[1,0], view_transform[1,1] = s, c
                                    # Add batch dimension
                                    view_transform = view_transform.unsqueeze(0)
                                    interventions['view'] = view_transform
                                    
                                # Add symmetry intervention if object is symmetric
                                if idx[0].item() in opt.sym_list:
                                    symmetry_transform = torch.eye(3).cuda()
                                    symmetry_transform[0,0] = -1  # Mirror across YZ plane
                                    # Add batch dimension
                                    symmetry_transform = symmetry_transform.unsqueeze(0) # Here the symmetry is incorrect
                                    interventions['symmetry'] = symmetry_transform # should the symmery and view have the same dimension? since it creates a 3\
                                    
                                # SCM refinement
                                pred_r, pred_t = refiner(new_points, emb, initial_pose, idx, interventions)
                                
                                # Use SCM loss instead of criterion_refine
                                features = refiner.geometric_features(new_points)
                                losses = scm_criterion(
                                    pred_r, pred_t, new_target, model_points, 
                                    new_points, features, interventions
                                )
                                losses['total_loss'].backward(retain_graph=True)
                                
                                dis = losses.get('pose_loss', torch.tensor(0.0))
                                # Update points for next iteration
                                new_points = refiner.do_intervention(new_points, 'view', view_transform) if 'view' in interventions else new_points
                        else:
                            loss.backward()
                        total_norm = 0.0
                        for name, param in refiner.named_parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2).item()
                                total_norm += param_norm ** 2
                        total_norm = total_norm ** 0.5
                        print(f"Gradient Norm for Refiner: {total_norm}")   

                        train_dis_avg += dis.item()
                        train_count += 1
                        avg_loss = train_dis_avg / train_count

                        pbar.update(1)
                        pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})

                        if train_count % opt.batch_size == 0:
                            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=10.0)
                            optimizer.step()
                            optimizer.zero_grad()

                        if train_count != 0 and train_count % 1000 == 0:
                            if opt.refine_start:
                                torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                            else:
                                torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
        pbar.close()
        # Log epoch summary
        train_dis_avg = train_dis_avg / train_count
        logger.info('Epoch {0} finished - Average Loss: {1:.6f}'.format(epoch, train_dis_avg))

        print('>>>>>>>>----------epoch {0} train finish | Avg Loss: {1:.6f}---------<<<<<<<<'.format(epoch, train_dis_avg))
        scheduler.step()



        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        test_pbar = tqdm(testdataloader, desc=f'Testing Epoch {epoch}')

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            if opt.refine_start:
                initial_pose = torch.cat([pred_r, pred_t], dim=2)
                initial_pose = initial_pose.permute(0,2,1)
                for ite in range(0, opt.iteration):
                    # Test without interventions
                    pred_r, pred_t = refiner(new_points, emb, initial_pose, idx, interventions)
                    features = refiner.geometric_features(new_points)
                    losses = scm_criterion(
                                    pred_r, pred_t, new_target, model_points, 
                                    new_points, features, interventions
                                )
                    dis = losses.get('pose_loss', torch.tensor(0.0))

            test_dis += dis.item()
            test_count += 1
            test_pbar.set_postfix({'Avg Loss': f'{test_dis/test_count:.6f}'})

        test_pbar.close()

        test_dis = test_dis / test_count
        print('>>>>>>>>----------Epoch {0} test complete | Avg Loss: {1:.6f}---------<<<<<<<<'.format(epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=opt.lr_rate)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'ycb_jpg':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, ext='jpg')
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'ycb_jpg':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, ext='jpg')
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()