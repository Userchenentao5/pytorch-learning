from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = 'dataset/train/ants/150801171_cd86f17ed8.jpg'
jpg_img = Image.open(image_path).convert('RGB')

writer = SummaryWriter(log_dir='logs')

tensor_transform = transforms.ToTensor()

# 像函数一样使用，传入参数即可使用
tensor_img = tensor_transform(jpg_img)

writer.add_image('tensor_image', tensor_img, 0)

writer.close()