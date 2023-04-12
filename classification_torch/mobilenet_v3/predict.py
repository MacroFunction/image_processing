import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import mobilenet_v3_small

def main():
    device = torch.device("cuda:0")

    data_transfrom = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_path = "test.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img = data_transfrom(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, 'r') as f:
        class_indixt = json.load(f)

    model = mobilenet_v3_small(num_classes=5, reduced_tail=False).to(device)
    weights_path = './MobileNetV3.pth'
    assert os.path.exists(weights_path), "file:'{}' does not exist.".format(weights_path)

    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    model.eval()
    with torch.no_grad():

        output = torch.squeeze((model(img.to(device)))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    predict_res = "class: {} prob: {:.3}".format(class_indixt[str(predict_cla)], predict[predict_cla].numpy())

    plt.title(predict_res)
    for i in range(len(predict)):
        print("class: {:10} prob: {:.3}".format(class_indixt[str(i)], predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()