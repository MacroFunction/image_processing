import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

def main():
    device = torch.device("cuda:0")

    data_transfrom = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_path = "test.jpg"
    assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
    img = Image.open(image_path)

    plt.imshow(img)

    img = data_transfrom(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=5).to(device)

    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    # print(predict[predict_cla].numpy())
    print_res = "class: {} prob:{:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.3}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()