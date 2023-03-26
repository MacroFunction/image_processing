import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #创建算法模型的实例
    net = LeNet()
    #加载在权重
    net.load_state_dict(torch.load('Lenet.pth'))

    #读取测试的图像
    im = Image.open('img.png')
    #对图像进行预处理
    im = transform(im)  # [C, H, W]
    #将处理的数据添加批次维度
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():   #在测试和验证种不进行梯度的计算
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
