from LapDepth.memory_config import configure_memory_optimization, gradient_accumulation_setup
from option import args, parser
import csv
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

#################### Distributed learning setting #######################
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
#########################################################################

import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from datasets.datasets_list import MyDataset, Transformer

import os
from utils import *

from logger import TermLogger, AverageMeter
from trainer import validate, train_net
from model import LS

import os

# minimal_cuda_test.py
import torch

import os

# Configure memory optimization (call before training starts)
configure_memory_optimization()


# Define main worker function to initialize each GPU process
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # Save current GPU ID to args object
    args.multigpu = False  # Initialize multi-GPU flag as False
    if args.distributed:  # Check if distributed training mode is used
        args.multigpu = True  # If distributed training, set multi-GPU flag to True
        args.rank = args.rank * ngpus_per_node + gpu  # Calculate global rank of current process
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print("==> gpu:", args.gpu, ", rank:", args.rank, ", batch_size:", args.batch_size, ", workers:", args.workers)
        torch.cuda.set_device(args.gpu)  # Set current GPU device
    elif args.gpu is None:
        print("==> DataParallel Training")
        args.multigpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    else:
        print("==> Single GPU Training")
        torch.cuda.set_device(args.gpu)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    save_path = save_path_formatter(args,
                                    parser)  # .replace(':','-')   Call function to generate save path (based on parameters and parser)
    # print(save_path)
    args.save_path = 'checkpoints' / save_path  # Set save path as subdirectory under 'checkpoints'
    if (args.rank == 0):
        print('=> number of GPU: ', args.gpu_num)
        print("=> information will be saved in {}".format(args.save_path))
    args.save_path.makedirs_p()  # Create directory for save path (including parent directories)
    torch.manual_seed(args.seed)  # Set PyTorch random seed to ensure reproducibility

    ##############################    Data loading part    ################################

    # Responsible for dataset preparation and loader configuration, setting appropriate parameters based on different datasets,
    # and configuring appropriate data samplers based on training mode (distributed or non-distributed).
    # It also sets epoch size and cuDNN optimization options.

    if args.dataset == 'KITTI':
        args.max_depth = 80.0  # If KITTI dataset, set max depth to 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    train_set = MyDataset(args,
                          train=True)  # Create training dataset instance using custom MyDataset class, pass parameters and train=True flag
    test_set = MyDataset(args, train=False)

    if (
            args.rank == 0):  # Check if current process is the main process (rank 0); in distributed training, only main process prints info
        print("=> Dataset: ", args.dataset)
        print("=> Data height: {}, width: {} ".format(args.height, args.width))
        print('=> train samples_num: {}  '.format(len(train_set)))
        print('=> test  samples_num: {}  '.format(len(test_set)))

    train_sampler = None  # Initialize training sampler as None (default to random sampling)
    test_sampler = None  # Initialize test sampler as None
    if args.distributed:  # Check if distributed training is used
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(  # Create training data loader
        # Training dataset, batch size, shuffle: if sampler is used, do not shuffle (controlled by sampler),
        # number of worker processes, pin_memory=True to accelerate GPU data transfer, sampler to use
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(  # Create validation data loader
        test_set, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.epoch_size == 0:  # Check if epoch_size is set to 0
        args.epoch_size = len(
            train_loader)  # If epoch_size is 0, set it to length of training loader, meaning one epoch will traverse the entire training set
    cudnn.benchmark = True  # Enable cuDNN auto-tuning to accelerate convolution operations; this improves performance when input sizes are fixed
    #########################################################################################

    ###################### Setting Network, Loss, Optimizer part ###################
    if (args.rank == 0):  # Check if current process is the main process (rank 0)
        print("=> creating model")
    # Instantiate model, passing parameters args; LS might be a lightweight depth estimation network
    Model = LS(args)
    ############################### Number of model parameters ##############################
    num_params_encoder = 0  # Initialize encoder parameter counter
    num_params_decoder = 0  # Initialize decoder parameter counter
    for p in Model.encoder.parameters():  # Iterate over all parameters of encoder
        num_params_encoder += p.numel()  # Accumulate encoder parameter count
    for p in Model.decoder.parameters():  # Iterate over all parameters of decoder
        num_params_decoder += p.numel()  # Accumulate decoder parameter count
    if (args.rank == 0):
        print("===============================================")
        print("model encoder parameters: ", num_params_encoder)
        print("model decoder parameters: ", num_params_decoder)
        print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
        trainable_params = sum([np.prod(p.shape) for p in Model.parameters() if
                                p.requires_grad])  # Count trainable parameters (only those requiring gradients)
        print("Total trainable parameters: {}".format(trainable_params))
        print("===============================================")
    ############################### apex distributed package wrapping ########################
    if args.distributed:
        if args.norm == 'BN':  # Check if batch normalization is used
            Model = nn.SyncBatchNorm.convert_sync_batchnorm(
                Model)  # Convert regular BN to synchronized BN (sync statistics across GPUs)
            if (args.rank == 0):
                print("=> use SyncBatchNorm")
        Model = Model.cuda(args.gpu)  # Move model to specified GPU
        Model = torch.nn.parallel.DistributedDataParallel(Model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)
        print("=> Model Initialized on GPU: {} - Distributed Training".format(args.gpu))
        enc_param = Model.module.encoder.parameters()  # Get encoder parameters (access original model via .module)
        dec_param = Model.module.decoder.parameters()  # Get decoder parameters
    elif args.gpu is None:  # Check if DataParallel (multi-GPU non-distributed) is used
        Model = Model.cuda()  # Move model to GPU
        Model = torch.nn.DataParallel(Model)  # Wrap model with DataParallel
        print("=> Model Initialized - DataParallel")
        enc_param = Model.module.encoder.parameters()
        dec_param = Model.module.decoder.parameters()
    else:  # Otherwise (single GPU training)
        Model = Model.cuda(args.gpu)  # Move model to specified GPU
        print("=> Model Initialized on GPU: {} - Single GPU training".format(args.gpu))
        enc_param = Model.encoder.parameters()
        dec_param = Model.decoder.parameters()

    ###########################################################################################

    ################################ Pretrained model loading #################################
    if args.model_dir != '':  # Check if pretrained model path is provided
        # Model.load_state_dict(torch.load(args.model_dir,map_location='cuda:'+args.gpu_num))
        Model.load_state_dict(torch.load(args.model_dir))  # Load pretrained model weights
        if (args.rank == 0):
            print('=> pretrained model is created')
    #############################################################################################

    ############################## Optimizer and loss function settings ##############################
    """
    Optimizer settings using Model.module (for wrapped models)
    optimizer = torch.optim.AdamW([{'params': Model.module.encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   {'params': Model.module.decoder.parameters(), 'weight_decay': 0, 'lr': args.lr}], eps=args.adam_eps)
    """

    # Create AdamW optimizer, set different parameters for encoder and decoder
    # Encoder: use weight decay and learning rate         Decoder: no weight decay (weight_decay=0), but same learning rate
    # Set epsilon value (adam_eps) for numerical stability
    optimizer = torch.optim.AdamW(
        [{'params': Model.encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
         {'params': Model.decoder.parameters(), 'weight_decay': 0, 'lr': args.lr}], eps=args.adam_eps)
    ##############################################################################################
    logger = None

    ####################################### Training part ##########################################

    if (args.rank == 0):
        print("training start!")

    loss = train_net(args, Model, optimizer, train_loader, val_loader, args.epochs, logger)

    if (args.rank == 0):
        print("training is finished")


if __name__ == '__main__':
    args.batch_size_dist = args.batch_size
    args.num_threads = args.workers
    args.world_size = 1
    args.rank = 0
    nodes = "127.0.0.1"
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        print("==> Distributed Training")
        mp.set_start_method('forkserver')

        print("==> Initial rank: ", args.rank)
        port = np.random.randint(10000, 10030)
        args.dist_url = 'tcp://{}:{}'.format(nodes, port)
        print("==> dist_url: ", args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None
        args.workers = 9
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)



