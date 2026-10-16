import torch
from torchvision import transforms
from torch import nn, optim
import torchvision.models as Model
from torchvision import transforms
from PIL import Image


def poly_adjust_lr(lr, itr, max_itr, power=0.9):
    now_lr = lr * (1 - itr / (max_itr + 1)) ** power
    return now_lr


def test1():
    lr = 0.1
    x = torch.tensor([1, 2, 3, 4.])
    x = x.detach()
    x.requires_grad = True
    print('x.requires_grad', x.requires_grad)
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Linear(16, 16),
        nn.Linear(16, 1),
    )
    model.eval()

    target = torch.tensor([1.])
    max_iter = 300
    with torch.no_grad():
        with torch.enable_grad():
            for i in range(max_iter):
                lr_now = lr
                # lr_now = poly_adjust_lr(lr, i, max_iter)
                y = model(x)
                criterion = nn.MSELoss()
                l = criterion(input=y, target=target)
                l.backward()
                print(f'loss:{l.item():.6f} | y:{y} | x.grad:{x.grad}')
                with torch.no_grad():
                    x -= lr_now * x.grad
                    x.grad.zero_()
    print(x)


def test2():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    lr = 0.1
    # x = torch.rand(1, 3, 128, 128)
    img = Image.open('/Users/chenlinwei/Desktop/goose_99.png').convert('RGB')
    img.show()

    x = transforms.ToTensor()(img)
    x = normalize(x)
    x = x.unsqueeze(dim=0)

    print(x)
    # transforms.ToPILImage()(x.squeeze()).show()
    # optimizer = optim.SGD(x.)
    x.requires_grad = True
    print(x.requires_grad)
    model = Model.vgg16(pretrained=True)
    model.eval()

    # model = nn.Sequential(
    #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(16),
    #     nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(8)
    #     nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
    # )

    # print(model.state_dict())
    target = torch.tensor([1])
    max_iter = 10
    for i in range(max_iter):
        lr_now = poly_adjust_lr(lr, i, max_iter)
        y = model(x)
        print(y.argmax(dim=1))
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        l = criterion(input=y, target=target)
        print(f'loss:{l.item():.6f}')
        l.backward()
        with torch.no_grad():
            # 用梯度下降更新权重
            x -= lr_now * x.grad
            x.grad.zero_()
    print(x)

    un_normalize = transforms.Normalize(mean=[1 / i for i in [0.485, 0.456, 0.406]],
                                        std=[1 / i for i in [0.229, 0.224, 0.225]])
    res_img = un_normalize(x.squeeze())
    res_img = transforms.ToPILImage()(res_img)
    res_img.show()


if __name__ == '__main__':
    test1()
