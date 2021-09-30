import torch
import numpy as np
import skimage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.optim as optim
from torch import nn
from torch.nn import functional

class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def loss_batch(model,loss_func,xb,yb,opt=None):
    loss = loss_func(model(xb),yb.long())
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return  loss.item(),len(xb)
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))
    return model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(42, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        x = functional.relu(self.hidden1(x))
        x = functional.relu(self.hidden2(x))
        x = functional.relu(self.hidden3(x))
        x = self.out(x)
        return x
if __name__ == '__main__':

    #随机生成数据，大小为10 * 20列
    source_data = np.load("x_l.npy")
    # 随机生成标签，大小为10 * 1列
    source_label = np.load("y_l.npy")
    source_data =source_data.reshape((-1,42))
    source_data =torch.tensor(source_data,dtype=torch.float32)
    source_label = torch.tensor(source_label)
    print(source_data)
    print(source_data.shape)
    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = GetLoader(source_data, source_label)
    datas = DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False, num_workers=0)
    net = MLP()
    print(net)
    train = TensorDataset(source_data,source_label)
    train = DataLoader(train,batch_size=10,shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    model = fit(30,net,loss_fun,optimizer,train,train)
    torch.save(model.state_dict(),"hand_control_model.pth")