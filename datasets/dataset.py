import numpy as np
from torch.utils.data import  Dataset,DataLoader
from torchvision.transforms import transforms
#  def cifar10_loader(data_path='/home/nick/Data/Datasets/torch', batch_size=128):
class Mydataset(Dataset):
    def __init__(self,data_path,label_path,transforms=None):
        self.datas = np.load(data_path)
        self.labels = np.load(label_path)
        self.transforms = transforms

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        data = data.reshape(32,32,1)   # 数据每一行读进来reshape成64x64x1
        #data = data.transpose(2,0,1)
        if self.transforms != None:
            data = self.transforms(data)
        label = np.argwhere(label==1)[0][0]
        return data, label

    def __len__(self):
        return len(self.datas)

def dataloader(train_data_path="./DANN18/train_X_t.npy",
               train_label_path="./DANN18/train_Y_t.npy",
               test_data_path="./DANN18/test_X_t.npy",
               test_label_path="./DANN18/test_Y_t.npy",
               batch_size=32,  # 后期看要不要改成128
               num_worker=4):
	train_transform = transforms.Compose([
		transforms.ToTensor()
	])
	test_transform = transforms.Compose([
		transforms.ToTensor()
	])
	train_set = Mydataset(train_data_path,train_label_path,train_transform)
	test_set = Mydataset(test_data_path,test_label_path,test_transform)
	original_set = Mydataset(train_data_path,train_label_path,test_transform)
	train_loader = DataLoader(train_set,batch_size=batch_size,num_workers=num_worker,shuffle=True,drop_last=True)
	test_loader = DataLoader(test_set,batch_size=batch_size,num_workers=num_worker,shuffle=False)
	original_loader = DataLoader(original_set,batch_size=batch_size,num_workers=num_worker,shuffle=False)
	return train_loader, test_loader, original_loader