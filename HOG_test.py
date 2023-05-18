import numpy as np
import cv2
import torch
from torch.nn import functional as F

def HOG(img, shift=1,mode='bilinear'):
	# img = cv2.imread('/mnt/cephfs/home/zhangshuhai/DiffPure/picture_diffusion_visual/original_picture.png')
	# img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).permute(0,3,1,2)
	# img = torch.randn(4,3,10,10)
	# shift = 1
	transform_matrix_left = torch.tensor([
				[1, 0, shift/img.shape[-1]],
				[0, 1 ,0]],device=img.device).expand((img.shape[0],2,3)) # 设B(batch size为1)
	transform_matrix_up = torch.tensor([
				[1, 0, 0],
				[0, 1 ,shift/img.shape[-1]]],device=img.device).expand((img.shape[0],2,3)) # 设B(batch size为1)
	grid_left = F.affine_grid(transform_matrix_left, # 旋转变换矩阵
						img.shape,align_corners=False)	# 变换后的tensor的shape(与输入tensor相同)
	grid_up = F.affine_grid(transform_matrix_up, # 旋转变换矩阵
						img.shape,align_corners=False)	# 变换后的tensor的shape(与输入tensor相同)
	output_left = F.grid_sample(img, # 输入tensor，shape为[B,C,W,H]
						grid_left, # 上一步输出的gird,shape为[B,C,W,H]
						mode=mode,align_corners=False) # 一些图像填充方法，这里我用的是最近邻
	output_up = F.grid_sample(img, # 输入tensor，shape为[B,C,W,H]
						grid_up, # 上一步输出的gird,shape为[B,C,W,H]
						mode=mode,align_corners=False) # 一些图像填充方法，这里我用的是最近邻
	output = (output_left**2+output_up**2).sqrt()
	
	return output

img = cv2.imread('/mnt/cephfs/home/zhangshuhai/DiffPure/picture_diffusion_visual/original_picture.png')
#在这里设置参数
winSize = (112,112)
blockSize = (56,56)
blockStride = (28,28)
cellSize = (28,28)
nbins = 9

#定义对象hog，同时输入定义的参数，剩下的默认即可
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

winStride = (8,8)
padding = (0,0)
test_hog = hog.compute(img, winStride, padding).reshape((-1,))
print("test_hog.shape",test_hog.shape)