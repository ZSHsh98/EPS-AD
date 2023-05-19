import torch
from torch.utils.data import Dataset 
import re
import pickle
import numpy as np
from PIL import Image
import os
import random


def sort_key(s):
	#sort_strings_with_embedded_numbers
	re_digits = re.compile(r'(\d+)')
	pieces = re_digits.split(s)  
	pieces[1::2] = map(int, pieces[1::2])  
	return pieces


def load_variavle(filename):
	f=open(filename,'rb')
	r=pickle.load(f)
	f.close()
	return r



class DatasetNPY(Dataset): 
	"""
	Clean dataset.
	Args:
		img_dirs: dir list to clean images from.
	"""

	def __init__(self, img_dirs,  transform = None):  
		self.img_dirs = img_dirs
		self.img_names = self.__get_imgnames__()
		self.transform = transform

	def __get_imgnames__(self):
		tmp = []
		images_name = os.listdir(self.img_dirs)
		images_name.sort(key=sort_key)
		for name in images_name:
			tmp.append(os.path.join(self.img_dirs, name))
		return tmp


	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		image_path   = self.img_names[idx]
		image        = np.load(image_path)
		image = image.astype(np.float32)

		# if self.transform:
		# 	image = self.transform(image)
		image = torch.from_numpy(image)

		return image
