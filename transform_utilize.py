from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter(log_dir='logs')

image_path = 'dataset/train/ants/403746349_71384f5b58.jpg'
jpg_img = Image.open(image_path).convert('RGB')


transform_toTense = transforms.ToTensor()

# __call__赋予的机制
tensor_img = transform_toTense(jpg_img)

print(tensor_img[0][0][0])

transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# output[channel] = (input[channel] - mean[channel]) / std[channel]
normalized_img = transforms_normalize(tensor_img)

print(normalized_img[0][0][0])
writer.add_image('Normalized Image', normalized_img, 0)

# size参数类型是Tuple[int, int]
transforms_resize = transforms.Resize((224, 224))
resized_img = transforms_resize(jpg_img)
print(resized_img)

resized_transformed_img = transform_toTense(resized_img)

writer.add_image('Resized Image', resized_transformed_img, 0)




writer.close()