import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from Function import train_model
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#数据加载
data_dir = 'D:\\Users\\ASUS\\Desktop\\良性癌症等检测'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True)
               for x in ['train', 'val']}
trainloader = dataloaders['train']
testloader = dataloaders['val']





# Model, Loss Function, and Optimizer
# from Models.ResNet32_Series import Get_LVPN_ResNet32
# model = Get_LVPN_ResNet32(2).to(device)
# from Models.ResNet32_Series import Get_ResNet32
# model = Get_ResNet32(2).to(device)
from Models.ResNet18_Series import Get_LVPN_ResNet18
model = Get_LVPN_ResNet18(2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)




#模型训练
train_model(model, trainloader, testloader,  criterion, optimizer, device, num_epochs=20)














