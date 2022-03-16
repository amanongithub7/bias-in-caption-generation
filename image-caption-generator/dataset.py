import torch
import os
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

class ImgCaptionDataset(torch.utils.data.Dataset):
  """
  ImgCaptionDataset returns an image, captions pair when its getitem method is 
  evoked.
  
  Required paramaters for initialization:
  - df: a pandas DataFrame containing an images column and a captions column
  - img_path: path of directory containing all the images
  - size: tuple of image size (channels, height, width)
  """
  def __init__(self, df, img_path, size):
    self.table = df
    self.path = img_path
    self.size = size
  
  def __getitem__(self, index):
    image_arr = (self.table.iloc[index]).to_numpy()
    image_name, caption = image_arr[0], image_arr[1]
    image = Image.open(os.path.join(self.path, image_name))
    image = image.resize((self.size[1], self.size[2]))

    # plt.imshow(image)
    # plt.show()
        
    # converting image to tensor and normalizing
    transformation = transforms.ToTensor()
    image = transformation(image)
    image = torch.reshape(image, (self.size[0], self.size[1], self.size[2]))
    # using emperical estimates to normalize image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    
    image = normalize(image)

    return image, torch.tensor(caption)

  def __len__(self):
    return len(self.table)