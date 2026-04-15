import numpy as np
import random
from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import time
from calculate_error import *
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import csv
import os
import imageio
from tqdm import tqdm
from path import Path
import torch
import numpy as np
from skimage.filters import sobel
from skimage.feature import canny
import warnings
warnings.filterwarnings(action='ignore')


def extract_edge_mask(rgb_np, method='canny', low_threshold=0.1, high_threshold=0.3):
    """
    rgb_np: numpy array (H,W,3) in range [0,1] or [0,255]
    returns: bool mask (H,W)
    """
    if rgb_np.max() > 1:
        rgb_np = rgb_np / 255.0
    gray = np.dot(rgb_np[..., :3], [0.299, 0.587, 0.114])
    if method == 'canny':
        edges = canny(gray, sigma=1.0, low_threshold=low_threshold, high_threshold=high_threshold)
    elif method == 'sobel':
        edges = sobel(gray) > 0.1
    else:
        raise ValueError("method must be 'canny' or 'sobel'")
    return edges


def scale_invariant_alignment(pred, gt, valid_mask):
    """
    pred, gt: torch.Tensor (1,1,H,W)
    valid_mask: torch.Tensor (1,1,H,W) bool
    returns: pred_aligned (same shape)
    """
    log_pred = torch.log(pred + 1e-6)
    log_gt = torch.log(gt + 1e-6)
    mean_log_pred = (log_pred * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    mean_log_gt = (log_gt * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    log_pred_aligned = log_pred - mean_log_pred + mean_log_gt
    return torch.exp(log_pred_aligned)


def compute_edge_rmse_and_f1(pred_depth, gt_depth, rgb_np, valid_mask_np,
                             edge_method='canny', rel_threshold=0.1, scale_invariant=True):
    """
    pred_depth, gt_depth: numpy arrays (H,W)
    rgb_np: numpy array (H,W,3)
    valid_mask_np: bool numpy array (H,W) indicating valid depth pixels
    returns: edge_rmse (float), boundary_f1 (float)
    """
    # Edge mask from RGB
    edge_mask = extract_edge_mask(rgb_np, method=edge_method)
    # Combine with valid depth mask
    edge_mask = edge_mask & valid_mask_np
    if edge_mask.sum() == 0:
        return 0.0, 0.0

    p = pred_depth.copy()
    g = gt_depth.copy()

    if scale_invariant:
        # Convert to torch for alignment
        p_t = torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0)
        g_t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0)
        vm_t = torch.from_numpy(edge_mask).bool().unsqueeze(0).unsqueeze(0)
        p_aligned = scale_invariant_alignment(p_t, g_t, vm_t)
        p = p_aligned.squeeze().cpu().numpy()

    # Edge RMSE
    diff = (p[edge_mask] - g[edge_mask]) ** 2
    edge_rmse = np.sqrt(diff.mean())

    # Boundary F1 (relative error threshold)
    rel_err = np.abs(p[edge_mask] - g[edge_mask]) / (g[edge_mask] + 1e-6)
    pred_edge = (rel_err < rel_threshold)
    tp = pred_edge.sum()
    total_edge = edge_mask.sum()
    fp = total_edge - tp  # incorrectly predicted as edge? Actually we treat all edge pixels as positive
    # For F1 we need precision and recall:
    #   precision = tp / (tp + fp)  (fp are false positives, here we have no negative edge ground truth)
    # Better definition: treat correct prediction as true positive, wrong as false negative.
    # Usually boundary F1: precision = TP/(TP+FP), recall = TP/(TP+FN)
    # FN = total_edge - TP
    fn = total_edge - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return edge_rmse, f1


def validate(args, val_loader, model, logger, dataset='KITTI'):
    batch_time = AverageMeter()
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    elif dataset == 'NYU':
        error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    errors = AverageMeter(i=len(error_names))
    edge_rmse_meter = AverageMeter()
    boundary_f1_meter = AverageMeter()

    model.eval()
    end = time.time()
    logger.valid_bar.update(0)

    for i, (rgb_data, gt_data, _) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        end = time.time()
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()

        # compute output with flip test
        input_img = rgb_data
        input_img_flip = torch.flip(input_img, [3])
        with torch.no_grad():
            _, output_depth = model(input_img)
            _, output_depth_flip = model(input_img_flip)
        output_depth_flip = torch.flip(output_depth_flip, [3])
        output_depth = 0.5 * (output_depth + output_depth_flip)
        batch_time.update(time.time() - end)

        # 标准误差计算
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth, crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth, crop=True)
        elif dataset == 'Make3D':
            err_result = compute_errors_Make3D(depth, output_depth)  # 注意这里 depth 未定义，应改为 gt_data? 原代码如此
        errors.update(err_result)

        # 边缘指标计算
        pred_np = output_depth.squeeze(1).cpu().numpy()  # (B, H, W)
        gt_np = gt_data.squeeze(1).cpu().numpy()  # (B, H, W)
        rgb_np = rgb_data.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
        valid_mask_np = (gt_np > 0)

        batch_edge_rmse = []
        batch_f1 = []
        for b in range(pred_np.shape[0]):
            edge_rmse, f1 = compute_edge_rmse_and_f1(
                pred_depth=pred_np[b],
                gt_depth=gt_np[b],
                rgb_np=rgb_np[b],
                valid_mask_np=valid_mask_np[b],
                edge_method='canny',
                rel_threshold=0.1,
                scale_invariant=True
            )
            batch_edge_rmse.append(edge_rmse)
            batch_f1.append(f1)

        edge_rmse_meter.update(np.mean(batch_edge_rmse), len(batch_edge_rmse))
        boundary_f1_meter.update(np.mean(batch_f1), len(batch_f1))

        logger.valid_bar.update(i + 1)
        if i % 10 == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(
                batch_time, errors.val[0], errors.avg[0]))
            logger.valid_writer.write('Edge RMSE: {:.4f}, Boundary F1: {:.4f}'.format(
                edge_rmse_meter.val, boundary_f1_meter.val))

    logger.valid_bar.update(len(val_loader))
    # 返回标准误差 + 两个边缘指标
    return errors.avg, error_names, edge_rmse_meter.avg, boundary_f1_meter.avg

def train_net(args,model, optimizer, dataset_loader,val_loader, n_epochs,logger):
    num = 0
    model_num = 0    
    
    data_iter = iter(dataset_loader)
    rgb_fixed, depth_fixed, _ = next(data_iter)
    depth_fixed = depth_fixed.cuda()
    
    save_dir = './' + args.dataset + '_LS_' + args.encoder + '_epoch' + str(n_epochs+5)
    
    if (args.rank == 0):
        print("Training for %d epochs..." % (n_epochs+5))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    

    H = args.height
    W = args.width

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    a1_acc_dir = str(train_loss_dir/'a1_acc_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    loss_pdf = "train_loss.pdf"
    rmse_pdf = "train_rmse.pdf"
    a1_pdf = "train_a1.pdf"        
    
    if args.dataset == "KITTI":
        # create mask for gradient loss
        y1_c,y2_c = int(0.40810811 * depth_fixed.size(2)), int(0.99189189 * depth_fixed.size(2))
        x1_c,x2_c = int(0.03594771 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))    ### Crop used by Garg ECCV 2016 
        y1,y2 = int(0.3324324 * H), int(0.99189189 * H)
        if (args.rank == 0):
            print(" - valid y range: %d ~ %d"%(y1,y2))
        crop_mask = depth_fixed != depth_fixed
        crop_mask[:,:,y1:y2,:] = 1
        crop_mask_a1 = depth_fixed != depth_fixed
        crop_mask_a1[:,:,y1_c:y2_c,x1_c:x2_c] = 1
    else:
        crop_mask = None

    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    a1_acc_list = []
    num_cnt = 0
    train_loss_cnt = 0

    n_iter = 0
    iter_per_epoch = len(dataset_loader)
    base_lr = args.lr
    end_lr = args.end_lr
    total_iter = n_epochs * iter_per_epoch
    ################ train mode ####################
    model.train()
    ################################################
    for epoch in tqdm(range(n_epochs+5)):
        #dataset_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)               # numpy 
        torch.manual_seed(epoch)            # cpu 
        torch.cuda.manual_seed(epoch)       # gpu 
        torch.cuda.manual_seed_all(epoch)   # gpu 
        ####################################### one epoch training #############################################
        for i, (rgb_data, gt_data, gt_dense) in enumerate(dataset_loader):

            # get the inputs
            inputs = rgb_data
            depths = gt_data
            inputs = inputs.cuda()
            depths = depths.cuda()
            inputs, depths = Variable(inputs), Variable(depths)
            if args.use_dense_depth is True:
                dense_depths = gt_dense
                dense_depths = dense_depths.cuda()
                dense_depths = Variable(dense_depths)
            
            '''Network loss'''
            # Feed-forward pass
            d_res_list, outputs = model(inputs)
            if args.lv6 is True:
                [lap6_pred, lap5_pred, lap4_pred, lap3_pred, lap2_pred, lap1_pred] = d_res_list
            else:
                [lap5_pred, lap4_pred, lap3_pred, lap2_pred, lap1_pred] = d_res_list
            ##################################### Valid mask definition ####################################
            # masking valied area
            valid_mask, final_mask = make_mask(depths, crop_mask, args.dataset)

            valid_out = outputs[valid_mask]
            valid_gt_sparse = depths[valid_mask]

            ###################################### scale invariant loss #####################################
            scale_inv_loss = scale_invariant_loss(valid_out, valid_gt_sparse)
            
            ###################################### gradient loss ############################################
            grad_epoch = 15 if args.dataset == 'KITTI' else 20
            if args.use_dense_depth is True:
                if epoch < grad_epoch:
                    gradient_loss = torch.tensor(0.).cuda()
                else:
                    gradient_loss = imgrad_loss(outputs, dense_depths, final_mask)
                    gradient_loss = 0.1*gradient_loss
            else:
                gradient_loss = torch.tensor(0.).cuda()

            loss = scale_inv_loss + gradient_loss

            # zero the parameter gradients and backward & optimize
            optimizer.zero_grad()
            loss.backward()

            if n_iter == total_iter:
                current_lr = end_lr
            else:
                current_lr = (base_lr - end_lr) * (1 - n_iter / total_iter) ** 0.5 + end_lr
                n_iter += 1

            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.step()
            print("iter_per_epoch:",iter_per_epoch)
            if ((i+1) % (iter_per_epoch//2) == 0) and (args.rank == 0):
                torch.save(model.state_dict(), save_dir+'/epoch_%02d_loss_%.4f_1.pkl' %(model_num+1,loss))
            if ((i+1) % args.print_freq == 0) and (args.rank == 0):
                print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                print("[%6d/%6d]  total: %.5f, gradient: %.5f, scale_inv: %.5f"%(n_iter, total_iter, loss.item(),gradient_loss.item(),scale_inv_loss.item()))
                total_loss = loss.item()                    
                rmse_loss = (torch.sqrt(torch.pow(valid_out.detach()-valid_gt_sparse,2))).mean()
                rmse_loss = rmse_loss.item()
                train_loss_cnt = train_loss_cnt + 1
                train_plot(args.save_path,total_loss, rmse_loss, train_loss_list, train_rmse_list, train_loss_dir,train_loss_dir_rmse,loss_pdf, rmse_pdf, train_loss_cnt,True)
                
                if args.val_in_train is True:
                    print("=> validate...")
                    a1_acc, rmse_test_loss, edge_rmse, boundary_f1 = validate(args, val_loader, model, logger, args.dataset)
                    validate_plot(args.save_path,a1_acc, a1_acc_list, a1_acc_dir,a1_pdf, train_loss_cnt,True)         

        if (args.rank == 0):
            print("=> learning decay... current lr: %.6f"%(current_lr))
            torch.save(model.state_dict(), save_dir+'/epoch_%02d_loss_%.4f_2.pkl' %(model_num+1,loss))
        model_num = model_num + 1
    
    return loss


if __name__ == "__main__":
    main()
