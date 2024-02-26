from __future__ import absolute_import
import argparse

def arg_parse():
	parser = argparse.ArgumentParser()

	# datasets parameters
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
	
	# training parameters
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--n_epoch', default=300, type=int)
	parser.add_argument('--warm_up_epoch', default=10, type=int)
	parser.add_argument('--train_batch', default=64, type=int)
	parser.add_argument('--lr', default=5e-3, type=float)
	parser.add_argument('--lr_min', default=5e-5, type=float)
	parser.add_argument('--weight_decay', default=0.0001, type=float)
	parser.add_argument('--momentum', default=0.0001, type=float)
	#parser.add_argument('--teacher_path', default=None, type=str)
	# transformer parameters
	parser.add_argument('--num_model', default=3, type=int)
	parser.add_argument('--d_model', default=256, type=int)
	parser.add_argument('--nhead', default=4, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--enc_layers', default=2, type=int)
	parser.add_argument('--dec_layers', default=2, type=int)
	parser.add_argument('--act_func', default='relu', type=str, help="activation function in transformer")
	parser.add_argument('--dropout', default=0.1, type=float)
	
	# resume trained model
	parser.add_argument('--resume', default='', type=str, help="path to the trained model")

	# others
	parser.add_argument('--save_dir', default='log', type=str)
	parser.add_argument('--save_model_name', default='model_best.pth', type=str)
	parser.add_argument('--random_seed', default=999, type=int)

	args = parser.parse_args()

	return args