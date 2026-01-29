import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

transform_compose = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10('dataset', train=True, download=True, transform=transform_compose)

test_dataset = torchvision.datasets.CIFAR10('dataset', train=False, download=True, transform=transform_compose)

# print(train_dataset)
# print(test_dataset)
#
# img, target = train_dataset[0]
# print(img)
# img.show()
# print(train_dataset.classes)
# print(target)

print(test_dataset[0])

writer = SummaryWriter(log_dir='logs')

for i in range(10):
    img, target = test_dataset[i]
    writer.add_image('test_dataset', img, i)


writer.close()