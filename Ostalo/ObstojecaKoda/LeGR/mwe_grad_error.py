import torch
import torch.nn as nn
from pytorchviz.torchviz.dot import make_dot


# original (used for teacher)
class DenseNet2D_original(nn.Module):
    def __init__(self):
        super(DenseNet2D_original, self).__init__()

        self.conv1 = nn.Conv2d(64, 25, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(25, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        self.x1 = self.conv1(x)
        self.x2 = self.conv2(self.x1)
        return self.x2

from utils.drivers import CrossEntropyLoss2d
def main():
    device = 'cuda'
    model = DenseNet2D_original()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    input_image = torch.rand((1, 64,640, 400))  #data shape torch.Size([4, 1, 640, 400])
    print(input_image.shape)
    data = input_image.to(device) # labels shape torch.Size([4, 640, 400])
    labels = torch.rand((1,640,400))
    target = labels.to(device).long()
    optimizer.zero_grad()
    print('doing forward')
    batch_outputs = model(data)

    criterion = CrossEntropyLoss2d()
    loss = criterion(batch_outputs, target)
    loss.backward()


    make_dot(batch_outputs.mean(), params=dict(model.named_parameters())).view()





if __name__ == '__main__':
    main()
