# Load the datase

import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize, RandomRotation
from PIL import Image

import torch
import os
def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
        img = Image.open(file_path).convert('RGB')
        return img


class VoxCeleb2(data.Dataset):
        def __init__(self, num_views, random_seed, dataset, additional_face=True, jittering=False, frames=-1, use_ab=False):
                if dataset == 1:
                        self.ids = np.load(os.environ['BASE_LOCATION'] + '/Datasets/large_voxceleb/train.npy').astype(np.str)
                if dataset == 2:
                        self.ids = np.load(os.environ['BASE_LOCATION'] + '/Datasets/large_voxceleb/val.npy').astype(np.str)
                if dataset == 3:
                        self.ids = np.load(os.environ['BASE_LOCATION'] + '/Datasets/large_voxceleb/test.npy').astype(np.str)
                self.rng = np.random.RandomState(random_seed)   
                self.num_views = num_views
                self.base_file = os.environ['VOX_CELEB_LOCATION'] + '/%s/'

                self.ids = [id for id in self.ids  if os.path.exists(self.base_file % id)]
                crop = 170
                if jittering == True:
                    precrop = crop + 20
                    crop = self.rng.randint(crop, precrop)
                    self.pose_transform = Compose([
                                RandomRotation(15),
                                Scale((256,256)),
                                               Pad((20,80,20,30)),
                                               CenterCrop(precrop), RandomCrop(crop),
                                               Scale((256,256))])
                    self.transform = Compose([
                                          RandomRotation(15),
                                          Scale((256,256)),
                                          Pad((20,80,20,30)),
                                          CenterCrop(precrop), RandomCrop(crop),
                                          Scale((256,256))])
                else:
                    precrop = crop
                    self.pose_transform = Compose([Scale((256,256)),
                                               Pad((20,80,20,30)),
                                               CenterCrop(precrop),
                                               Scale((256,256))])
                    self.transform = Compose([Scale((256,256)),
                                          Pad((20,80,20,30)),
                                          CenterCrop(precrop),
                                          Scale((256,256))])
        
                self.use_ab = use_ab

        def __len__(self):
                return len(self.ids)

        def __getitem__(self, index):
                face1 = self.get_blw_item(index) 
                face2 = self.get_blw_item(self.rng.randint(self.__len__())) 
                return {'face1' : face1, 'face2' : face2}
        
        def get_blw_item(self, index):
                # Load the images
                imgs = [0] * (self.num_views)

                img_track = [d for d in os.listdir(self.base_file % self.ids[index]) if os.path.isdir(self.base_file % self.ids[index] + '/' + d)]
                img_track_t = []
                while(len(img_track_t) == 0):
                        img_video = img_track[self.rng.randint(len(img_track))]
                
                        img_track_t = []
                        img_track_t = [img_video + '/' + d for d in os.listdir(self.base_file % self.ids[index] + '/' + img_video) if not(d == 'VISITED')]
                img_track = img_track_t[self.rng.randint(len(img_track_t))]

                img_faces = [d for d in os.listdir(self.base_file % self.ids[index] + '/' + img_track) if d[-4:] == '.jpg']

                if self.num_views > len(img_faces):
                        img_index = self.rng.choice(range(len(img_faces)), self.num_views, replace=True)
                else:
                        img_index = self.rng.choice(range(len(img_faces)), self.num_views, replace=False)

                for i in range(0, self.num_views):
                        img_name = self.base_file % self.ids[index] + '/' + img_track + '/' + img_faces[img_index[i]]
                        imgs[i] = load_img(img_name)
                        img = self.transform(imgs[i])
                        if self.use_ab:
                                img = np.asarray(img)
                                img_lab = rgb2lab(img)
                                img_lab = (img_lab + 128) / 255
                                img_ab = img_lab[:, :, 1:3].astype(np.float32)
                                img = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                        else:
                                img = ToTensor()(img)
                        imgs[i] = img

                return imgs

