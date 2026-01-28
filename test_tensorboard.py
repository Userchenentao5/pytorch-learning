import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
image_path = 'dataset/train/ants/150801171_cd86f17ed8.jpg'
img = Image.open(image_path)
img_array = np.array(img)
writer.add_image('train-ants', img_array, 2, dataformats="HWC")


for i in range(100):
    writer.add_scalar('y=2x', i * 2, i)

writer.close()