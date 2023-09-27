#reference
#https://blog.csdn.net/qq_36693723/article/details/130332573?ops_request_misc=&request_id=&biz_id=102&utm_term=gan%20pytorch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-130332573.142^v94^insert_down28v1&spm=1018.2226.3001.4187

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #for GPU
device =  torch.device("mps" if torch.backends.mps.is_available() else "cpu")   #for M2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=True)

class Generator(nn.Module):
    '''定义生成器'''

    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        '''
        input_dim: int 随机取样的噪音维度
        hidden_dim: int 隐藏层维度
        '''
        self.input_dim = input_dim  #随机取样的噪音维度
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 28*28)

    def forward(self, x):  #x:(batch_size, input_dim)
        x = self.fc1(x) #x:(batch_size, hidden_dim)
        x = F.relu(x)
        x = self.fc2(x) #x:(batch_size, 28*28)
        x = F.tanh(x) # 将输出值映射到 [-1, 1] 的范围
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    '''定义判别器'''
    
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        '''
        hidden_dim: int 隐藏层维度
        '''
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):   #x:(batch_size, 1, 28, 28)
        x = x.view(-1, 28*28)   #x:(batch_size, 28*28)
        x = self.fc1(x) #x:(batch_size, hidden_dim)
        x = F.relu(x)
        x = self.fc2(x) #x:(batch_size, 1)
        x = F.sigmoid(x)    #将输出值映射到 [0, 1] 的范围，表示判别为真实图像的概率
        return x

# 定义超参数
lr = 0.001
input_dim = 100
hidden_dim= 300
num_epochs = 50

generator = Generator(input_dim, hidden_dim).to(device)
discriminator = Discriminator(hidden_dim).to(device)

# 定义优化器和损失函数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

criterion = nn.BCELoss()

if __name__ == '__main__':
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(data_loader):
            # 将真实图像传递给设备
            real_images = real_images.to(device)

            # 定义真实标签为 1，假的标签为 0
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            # 训练生成器
            z = torch.randn(real_images.size(0), input_dim).to(device)  # 生成随机的噪音 z
            fake_images = generator(z)  # 生成 fake_images
            fake_output = discriminator(fake_images)    # 将 fake_images 传递给判别器，得到输出 fake_output

            loss_G = criterion(fake_output, real_labels)    # 计算生成器的损失值 loss_G,需要将 fake_output 与真实标签 real_labels 进行比较

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器
            real_output = discriminator(real_images)    # 将真实图像 real_images 通过判别器得到输出 real_output
            loss_D_real = criterion(real_output, real_labels)   # 计算判别器对真实图像的损失值 loss_D_real,需要将 real_output 与真实标签 real_labels 进行比较
            fake_output = discriminator(fake_images.detach())   # 计算判别器对生成器生成的 fake_images 的损失值 loss_D_fake,需要将 fake_output 与假的标签 fake_labels 进行比较
            loss_D_fake = criterion(fake_output, fake_labels)

            loss_D = loss_D_real + loss_D_fake  # 计算判别器总损失值 loss_D，即 loss_D_real 和 loss_D_fake 的和

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # 输出损失值
            # 每100次迭代输出一次
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, len(data_loader), loss_D.item(), loss_G.item()))
                
    # 生成一些测试数据
    z = torch.randn(16, input_dim).to(device)   # 随机生成一些长度为 input_dim 的、位于设备上的向量 z
    fake_images = generator(z).detach().cpu()   # 使用生成器从 z 中生成一些假的图片

    # 显示生成的图像
    # 创建一个图形对象，大小为 4x4 英寸
    fig = plt.figure(figsize=(4, 4))

    # 在图形对象中创建4x4的网格，以显示输出的16张假图像
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(fake_images[i][0], cmap='gray')
        plt.axis('off')

    plt.savefig('./generated_mnist')
