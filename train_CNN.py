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
	#file_out = open("/home/gongyong/zhangsirui/Deepfake-Detection/train/output_train.txt","w")
	#file_out = open("/home/gongyong/zhangsirui/Deepfake-Detection/train/Face2Face/output_train_Face2Face.txt","w")
	#file_out = open("/home/gongyong/zhangsirui/Deepfake-Detection/train/FaceSwap/output_train_FaceSwap.txt","w")
	file_out = open("/home/gongyong/zhangsirui/Deepfake-Detection/train/Neural/output_train_Neural_1.txt","w")
	args = parse.parse_args()
	name = args.name
	continue_train = args.continue_train
	train_list = args.train_list
	train_list_sample = args.train_list_sample
	val_list = args.val_list
	val_list_sample= args.val_list_sample
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('./output', name)
	
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	#torch.backends.cudnn.benchmark=True将会让程序在开始时花费一点额外时间，
	#为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
	torch.backends.cudnn.benchmark=True
	
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	
	if continue_train:
		model.load_state_dict(torch.load(model_path))
	model = model.cuda()

	criterion = nn.CrossEntropyLoss()

# 构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
# betas:用于计算梯度的平均和平方的系数,eps:为了提高数值稳定性而添加到分母的一个项
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	
# 调整学习率的一种机制——StepLR机制
# class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
# step_size ：每训练step_size个epoch，更新一次参数
# gamma ：更新lr的乘法因子 即lr、0.5lr、0.25lr、0.125lr......
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()  #用于存储模型的参数
	best_acc = 0.0

	#iteration : 使用一个batch数据对模型进行一次参数更新的过程
	#epoch : 使用训练集的全部数据对模型进行一次完整的训练
	iteration = 0
	for epoch in range(epoches):
		
		#解析出来数据集的内容
		train_dataset = MyDataset(txt_path_0=train_list, txt_path_1=train_list_sample, model=0,transform=xception_default_data_transforms['train'])
		train_dataset_size = len(train_dataset)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
		
		print('Epoch {}/{}'.format(epoch+1, epoches))
		file_out.write('Epoch {}/{}'.format(epoch+1, epoches) + "\n")
		print('-'*10)
		file_out.write('-'*10 + "\n")
		# model.train()的作用是启用 Batch Normalization 和 Dropout。
		# model.train()是保证BN层能够用到每一批数据的均值和方差。
		# 对于Dropout , model.train()是随机取一部分网络连接来训练更新参数。
		model.train()
		
		train_loss = 0.0
		train_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0
		
		for (image, labels) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			
			image = image.cuda()	
			labels = labels.cuda()
			
			optimizer.zero_grad()      #梯度置0
			
			#outputs为 dropout 后的
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)  #dropout 后每行最大数的序列数
			
			loss = criterion(outputs, labels)
			loss.backward()   #计算梯度
			optimizer.step()   #更新参数
			
			iter_loss = loss.data.item()
			train_loss += iter_loss
			
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			train_corrects += iter_corrects
			
			iteration += 1
			
			if not (iteration % 20):
				print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
				file_out.write('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size) + "\n")
		
		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size
		print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
		file_out.write('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc) + "\n")
        #-----------------------------------------------------------------------------------------------------------------------
		# 将网络层设置为验证的模式
		# 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
		# pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
		val_dataset = MyDataset(txt_path_0=val_list, txt_path_1=val_list_sample , model=1 ,transform=xception_default_data_transforms['val'])
		val_dataset_size = len(val_dataset)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

		model.eval()
		with torch.no_grad():    #停止梯度计算，禁止反向传播
			for (image, labels) in val_loader:   #验证集
				image = image.cuda()
				labels = labels.cuda()
				
				outputs = model(image)
				_, preds = torch.max(outputs.data, 1)
				
				loss = criterion(outputs, labels)
				
				val_loss += loss.data.item()
				val_corrects += torch.sum(preds == labels.data).to(torch.float32)
			
			epoch_loss = val_loss / val_dataset_size
			epoch_acc = val_corrects / val_dataset_size
			
			print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
			file_out.write('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc) + "\n")
			
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()  #更新最好模型的模型参数
		
		scheduler.step()     #每个epoch用一次进行学习率参数的更新
		#if not (epoch % 40):
		torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name)) #保存每个epoch的模型参数
	
	print('Best val Acc: {:.4f}'.format(best_acc))
	file_out.write('Best val Acc: {:.4f}'.format(best_acc) + "\n")
	
	model.load_state_dict(best_model_wts)
	torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))     #保存最优模型的模型参数
	file_out.close()


if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='Neural_c23_1') #Face2Face_c23 || FaceSwap_c23 ||Neural_c23
	parse.add_argument('--train_list', '-tl' , type=str, default = '')
	parse.add_argument('--train_list_sample', '-tm' , type=str, default = '')
	parse.add_argument('--val_list', '-vl' , type=str, default = '')
	parse.add_argument('--val_list_sample', '-vm' , type=str, default = '')
	parse.add_argument('--batch_size', '-bz', type=int, default=64)
	parse.add_argument('--epoches', '-e', type=int, default='10')
	parse.add_argument('--model_name', '-mn', type=str, default='Neural_c23.pkl')
	parse.add_argument('--continue_train', type=bool, default=True)
	parse.add_argument('--model_path', '-mp', type=str, default='')
	main()

