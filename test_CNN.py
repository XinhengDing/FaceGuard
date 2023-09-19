import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
def main():
	file_out = open("/home/gongyong/zhangsirui/Deepfake-Detection/train/Neural/output_test_Neural.txt","w")
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True
	
	test_dataset = MyDataset(txt_path_0=test_list,txt_path_1=None ,model=2, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	
	test_dataset_size = len(test_dataset)
	time = test_dataset_size/batch_size
	corrects = 0
	acc = 0
	iteration = 0
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	print(time)
	file_out.write('Iteration total: {}'.format(time))

	model.eval()
	with torch.no_grad():
		for (image, labels) in test_loader:
			image = image.cuda()
			labels = labels.cuda()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			corrects += torch.sum(preds == labels.data).to(torch.float32)
			iteration += 1
			print('Iteration {} Acc {:.4f}'.format(iteration,torch.sum(preds == labels.data).to(torch.float32)/batch_size))
			file_out.write('Iteration {} Acc {:.4f}'.format(iteration,torch.sum(preds == labels.data).to(torch.float32)/batch_size) + "\n")
		acc = corrects / test_dataset_size
		print('Test Acc: {:.4f}'.format(acc))
		file_out.write('Test Acc: {:.4f}'.format(acc) + "\n")



if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=64)
	parse.add_argument('--test_list', '-tl', type=str, default='/home/gongyong/zhangsirui/Deepfake-Detection/train/Neural/1c23_test_orig_Neural.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='/home/gongyong/zhangsirui/Deepfake-Detection/output/Neural_c23_1/best.pkl')
	
	main()

	