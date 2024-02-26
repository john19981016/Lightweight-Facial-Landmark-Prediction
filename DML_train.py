import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import parser
import models as Models
import data
import utility
import ConvNext

def save_model(model, save_path):
	torch.save(model.state_dict(), save_path)

def adjust_learning_rate(optimizer, total_epoch, cur_epoch, warm_up_epoch, lr_max, lr_min):
	# cosine decay lr with linear warmup
	if cur_epoch <= warm_up_epoch:
		lr = float(cur_epoch) * lr_max / warm_up_epoch
	else:
		lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos( (cur_epoch-warm_up_epoch)/(total_epoch-warm_up_epoch)*math.pi ))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

if __name__=='__main__':

	args = parser.arg_parse()

	torch.set_default_dtype(torch.float32)
	
	''' create directory to save trained model and other info '''
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	
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

	print('===> prepare model ...')
	#
	num_model=args.num_model
	models=[]
	for i in range(num_model): # init models randomly
		#model = Models.BaseModel(args).to(device)
		model = ConvNext.ReducedConvNeXt(in_chans=3, num_classes=args.n_landmark*2).to(device)
		model.apply(init_weights)
		models.append(model)


	summary(models[0], ( 3, 384, 384))
	#torch.save(model.state_dict(), args.save_model_name)

	''' define loss '''
	criterion = utility.SmoothWingLoss()
	criterion_KD = nn.KLDivLoss()
	''' setup optimizer '''
	#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	optimizers=[]
	for i in range(num_model):
		optimizers.append(optim.AdamW(models[i].parameters(), lr=args.lr, weight_decay=args.weight_decay))

	''' setup tensorboard '''
	#writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

	print('===> start training ...')
	best_nmes=[]
	best_epochs=[]
	for i in range(num_model):
		best_nmes.append(1e10)
		best_epochs.append(0)
	
	for epoch in tqdm(range(1, args.n_epoch+1)):

		for i in range(num_model):
			adjust_learning_rate(optimizers[i], args.n_epoch, epoch, args.warm_up_epoch, args.lr, args.lr_min)
		train_loss = 0
		for (imgs, gts) in tqdm((train_loader)):
			imgs = imgs.to(device); gts = gts.to(device)
			batch_size = imgs.size(0)

			for i in range(num_model): # reset the state
				optimizers[i].zero_grad()
				models[i].eval()
			
			for j in range(num_model): # train the j-th model
				models[j].train()
				preds, feats = models[j](imgs)
				preds = preds.view(batch_size, args.n_landmark, 2).contiguous()
				loss = criterion(preds, gts)
				for k in range(num_model): # compute the mutual loss
					if k==j:
						continue
					with torch.no_grad():
						preds_other, feats_other = models[k](imgs)
					loss += criterion_KD(feats, feats_other)
				loss.backward()
				optimizers[j].step()
				train_loss += loss.item()
				models[j].eval()
		train_loss /= len(train_loader)
		for i in range(num_model):
			nme = utility.evaluate(models[i], val_loader, device)
			if nme < best_nmes[i]:
				torch.save(models[i].state_dict(), str(i)+args.save_model_name)
				best_nmes[i] = nme
				best_epochs[i] = epoch

		print('Epoch[{}/{}] loss = {:.5f}'.format(epoch, args.n_epoch, train_loss))
		for i in range(num_model):
			print('Best NME_{} = {:.6f} (at epoch {})'.format(i,best_nmes[i], best_epochs[i]))

	print('-'*50)
	for i in range(num_model):
		print('Best NME_{} = {:.6f} (at epoch {})'.format(i,best_nmes[i], best_epochs[i]))

