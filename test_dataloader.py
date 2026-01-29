import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_dataset = torchvision.datasets.CIFAR10('dataset', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

data_loader = DataLoader(test_dataset, 64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter(log_dir='dataloader')

# enumerate使序列带有索引

for epoch in range(2):
    for step, data in enumerate(data_loader):
        imgs, targets = data
        writer.add_images('Epoch:{}'.format(epoch), imgs, step)

writer.close()