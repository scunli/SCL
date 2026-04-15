import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
import random
import torch
import time
import cv2
from PIL import ImageFile
from transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,Normalize
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

#判断输入对象是否为PIL图像类型
def _is_pil_image(img):
    return isinstance(img, Image.Image)

class MyDataset(data.Dataset):
    #初始化数据集，根据 args 配置和是否训练模式设置路径、参数等
    def __init__(self, args, train=True, return_filename = False):
        #是否使用密集深度图（dense depth）
        self.use_dense_depth = args.use_dense_depth

        if train is True:  #检查当前是否处于训练模式
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti #设置训练数据文件路径为 KITTI 训练文件路径。
                self.angle_range = (-1, 1)  #设置随机旋转的角度范围为 -1 到 1 度（用于数据增强）
                self.depth_scale = 256.0 #设置深度值的缩放因子为 256.0（KITTI 数据集的深度值通常需要除以 256 来得到真实深度）
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 416 #设置图像高度为 416 像素（NYU 数据集的特定设置）
                args.width = 544
        else: #如果不是训练模式（即测试模式）
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti  #设置测试数据文件路径为 KITTI 测试文件路径
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        self.train = train  #保存训练/测试模式标志到实例变量
        self.transform = Transformer(args) #创建数据转换器实例，根据数据集类型和模式配置相应的数据增强和预处理操作
        self.args = args  #保存参数对象到实例变量
        self.return_filename = return_filename #保存是否返回文件名的标志到实例变量
        with open(self.datafile, 'r') as f: #打开数据文件（包含图像路径列表）
            self.fileset = f.readlines()  #读取文件中的所有行，每行代表一个数据样本的路径信息
        self.fileset = sorted(self.fileset) #对文件列表进行排序，确保数据加载的顺序一致性

    def __getitem__(self, index):
        #获取指定索引处的文件路径字符串，并使用 split() 方法将其分割成列表。
        # 例如，如果 self.fileset[index] 是 "path/to/image.png path/to/depth.png"，
        # 则 divided_file 会成为 ["path/to/image.png", "path/to/depth.png"]。
        divided_file = self.fileset[index].split()
        if self.args.dataset == 'KITTI':
            #从第一个文件路径中提取日期部分
            date = divided_file[0].split('/')[0] + '/'

        # Opening image files.   rgb: input color image, gt: sparse depth map
        #根据数据集类型和模式（训练/测试）加载和处理图像及深度图数据。
        # 在训练模式下，会应用随机旋转作为数据增强；在测试模式下，会提取文件名信息用于后续的结果保存或评估。
        rgb_file = self.args.data_path + '/' + divided_file[0] #构建 RGB 图像文件的完整路径，将数据路径与文件列表中的第一个路径拼接
        rgb = Image.open(rgb_file) #使用 PIL 库打开 RGB 图像文件
        gt = False
        gt_dense = False
        if (self.train is False):
            divided_file_ = divided_file[0].split('/') #将 RGB 文件路径按 '/' 分割成列表，用于提取文件名信息
            if self.args.dataset == 'KITTI':
                filename = divided_file_[1] + '_' + divided_file_[4]
            else:
                filename = divided_file_[0] + '_' + divided_file_[1]
            
            if self.args.dataset == 'KITTI':
                # Considering missing gt in Eigen split
                if divided_file[1] != 'None':
                    #构建稀疏深度图文件的完整路径
                    gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        #构建密集深度图文件的完整路径
                        gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
                        gt_dense = Image.open(gt_dense_file)
                else:
                    pass
            elif self.args.dataset == 'NYU':
                #构建稀疏深度图文件的完整路径
                gt_file = self.args.data_path + '/' + divided_file[1]
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    #构建密集深度图文件的完整路径
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                    gt_dense = Image.open(gt_dense_file)
        else: #如果不是测试模式（即训练模式）
            #生成一个随机旋转角度，用于数据增强
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                #构建稀疏深度图文件的完整路径
                gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                if self.use_dense_depth is True:
                    #构建密集深度图文件的完整路径
                    gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
            elif self.args.dataset == 'NYU':
                #构建稀疏深度图文件的完整路径
                gt_file = self.args.data_path + '/' + divided_file[1]
                if self.use_dense_depth is True:
                    #构建密集深度图文件的完整路径
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
            
            gt = Image.open(gt_file)            
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file) 
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        # 裁剪图像尺寸使其能被16整除（通常是因为神经网络中的下采样操作需要）
        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216)//2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            if self.train is True:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 0
                bound_right = 640
                bound_top = 0
                bound_bottom = 480

        #对图像和深度图进行裁剪、归一化和转换处理，确保它们符合模型的输入要求，并根据需要返回处理后的数据和文件名

        # 接下来进行裁剪和归一化操作，RGB 范围归一化到 (0,1)，深度范围归一化到 (0, max_depth)。
        if (self.args.dataset == 'NYU' and (self.train is False) and (self.return_filename is False)):
            rgb = rgb.crop((40,42,616,474))
        else:
            rgb = rgb.crop((bound_left,bound_top,bound_right,bound_bottom))
            
        rgb = np.asarray(rgb, dtype=np.float32)/255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left,bound_top,bound_right,bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32))/self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        if self.use_dense_depth is True:
            if _is_pil_image(gt_dense):
                gt_dense = gt_dense.crop((bound_left,bound_top,bound_right,bound_bottom))
                gt_dense = (np.asarray(gt_dense, dtype=np.float32))/self.depth_scale
                gt_dense = np.expand_dims(gt_dense, axis=2)
                gt_dense = np.clip(gt_dense, 0, self.args.max_depth)
                gt_dense = gt_dense * (gt.max()/gt_dense.max())

        rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)

        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        else:
            return rgb, gt, gt_dense

    def __len__(self):
        return len(self.fileset)

class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2),brightness_mult_range=(0.75, 1.25)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
