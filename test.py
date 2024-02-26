import os
import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary

import parser
import models
import data
import utility
import ConvNext


if __name__=='__main__':

	args = parser.arg_parse()

	torch.set_default_dtype(torch.float32)
	
	''' setup random seed '''
	myseed = args.random_seed
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(myseed)
	torch.manual_seed(myseed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(myseed)

	device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

	print('===> prepare dataloader ...')
	train_loader = torch.utils.data.DataLoader(data.DATA(args.data_dir, mode='train'),
											   batch_size=args.train_batch, 
											   num_workers=args.workers,
											   shuffle=True,
											   pin_memory=True)
	val_loader   = torch.utils.data.DataLoader(data.DATA(args.data_dir, mode='val'),
											   batch_size=args.train_batch,
											   num_workers=args.workers,
											   shuffle=False,
											   pin_memory=True)
	test_loader  = torch.utils.data.DataLoader(data.DATA(args.data_dir, mode='test'),
											   batch_size=args.train_batch,
											   num_workers=args.workers,
											   shuffle=False,
											   pin_memory=True)

	print('===> prepare model ...')
	model = ConvNext.ReducedConvNeXt(in_chans=3, num_classes=args.n_landmark*2, depths=[3, 9, 3], dims=[96, 192, 384], drop_path_rate=0.1)
	#model = models.BaseModel(args)

	model.load_state_dict(torch.load(args.save_model_name, map_location=device))
	model.to(device)

	print('===> start testing ...')
	trainset_nme = utility.evaluate(model, train_loader, device)
	valset_nme = utility.evaluate(model, val_loader, device)

	print('Train set nme = {:.5f} | Val set nme = {:.5f}'.format(trainset_nme, valset_nme))

	utility.generate_test_txt(model, test_loader, device)