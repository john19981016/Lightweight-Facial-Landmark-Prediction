import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

MEAN =[0.485, 0.456, 0.406]     #[1, 1, 1]
STD = [0.229, 0.224, 0.225]     #[1, 1, 1]

def generate_transformed_pairs(image, landmarks):
    # perform random horizontal flip and random rotation
	# landmarks need to be transformed as well
    new_landmarks = landmarks.copy()
    # random hflip
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        new_landmarks[:,0] = image.shape[2] - new_landmarks[:,0]
        idx_order = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, \
                        26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, \
                        45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, \
                        59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]
        new_landmarks = new_landmarks[idx_order]
    # random rotate
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = transforms.functional.rotate(image, angle)
        radian = -angle * np.pi / 180
        rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)],
                                    [np.sin(radian), np.cos(radian)]])
        new_landmarks[:,0] = new_landmarks[:,0] - image.shape[2]/2
        new_landmarks[:,1] = new_landmarks[:,1] - image.shape[1]/2
        new_landmarks = np.transpose(np.matmul(rotation_matrix, np.transpose(new_landmarks)))
        new_landmarks[:,0] = new_landmarks[:,0] + image.shape[2]/2
        new_landmarks[:,1] = new_landmarks[:,1] + image.shape[1]/2
    
    return image, new_landmarks



class DATA(Dataset):
	def __init__(self, data_dir, mode='train'):

		self.mode = mode
		if self.mode == 'train':
			self.data_dir = os.path.join(data_dir, 'synthetics_train')
		elif self.mode == 'val':
			self.data_dir = os.path.join(data_dir, 'aflw_val')
		else:
			self.data_dir = os.path.join(data_dir, 'aflw_test')

		if self.mode != 'test':
			with open(os.path.join(self.data_dir, 'annot.pkl'), 'rb') as f:
				self.img_name, self.landmarks = pickle.load(f)
			self.landmarks = np.asarray(self.landmarks)
		else:
			self.img_name = os.listdir(self.data_dir)
			self.landmarks = np.array([])

		if self.mode == 'train':
			self.transform = transforms.Compose([
								#transforms.Resize(224),
								transforms.GaussianBlur((3, 3), sigma=(0.5, 2.0)),
								transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),	# sharpness_factor = 0 gives blurred image
								transforms.RandomGrayscale(p=0.2),
								transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
								transforms.ToTensor(), 		# (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
								transforms.Normalize(MEAN, STD)
							])
		else:
			self.transform = transforms.Compose([
								#transforms.Resize(224),
								transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
								transforms.Normalize(MEAN, STD)
							])

	def __len__(self):
		return len(self.img_name)

	def __getitem__(self, idx):
		img_path = os.path.join(self.data_dir, self.img_name[idx])
		img = Image.open(img_path).convert('RGB')
		
		if self.mode == 'train':
			return generate_transformed_pairs(self.transform(img), self.landmarks[idx])
		elif self.mode == 'val':
			return self.transform(img), self.landmarks[idx]
		else:
			return self.transform(img), self.img_name[idx]


class DATA_DANN(Dataset):
	def __init__(self, data_dir, mode='train'):

		self.mode = mode
		self.train_data_dir = os.path.join(data_dir, 'synthetics_train')
		self.val_data_dir = os.path.join(data_dir, 'aflw_val')

		with open(os.path.join(self.train_data_dir, 'annot.pkl'), 'rb') as f:
			self.train_img_names, self.train_landmarks = pickle.load(f)
		self.train_landmarks = np.asarray(self.train_landmarks)

		with open(os.path.join(self.val_data_dir, 'annot.pkl'), 'rb') as f:
			self.val_img_names, self.val_landmarks = pickle.load(f)
		self.val_landmarks = np.asarray(self.val_landmarks)

		if self.mode == 'train':
			self.transform = transforms.Compose([
								#transforms.Resize(224),
								transforms.GaussianBlur((3, 3), sigma=(0.5, 2.0)),
								transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),	# sharpness_factor = 0 gives blurred image
								transforms.RandomGrayscale(p=0.2),
								transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
								transforms.ToTensor(), 		# (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
								transforms.Normalize(MEAN, STD)
							])
		else:
			self.transform = transforms.Compose([
								#transforms.Resize(224),
								transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
								transforms.Normalize(MEAN, STD)
							])

	def __len__(self):
		return len(self.train_img_names)

	def __getitem__(self, idx):
		img_path = os.path.join(self.train_data_dir, self.train_img_names[idx])
		train_img = Image.open(img_path).convert('RGB')

		val_idx = idx % len(self.val_img_names)
		img_path = os.path.join(self.val_data_dir, self.val_img_names[val_idx])
		val_img = Image.open(img_path).convert('RGB')

		return self.transform(train_img), self.train_landmarks[idx], self.transform(val_img), self.val_landmarks[val_idx]