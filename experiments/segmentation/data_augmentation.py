import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transform

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes = 1, length = 50):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class PatchGaussian(object):
    """Adds noise to randomly selected patches
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        gaussian_sigma(float): in range [0,1], standard deviations of Gaussian noise
    """
    def __init__(self, n_holes = 1, length = 50, gaussian_sigma = 0.5):
        self.n_holes = n_holes
        self.length = length
        self.gaussian_sigma = gaussian_sigma
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image applied Patch Gaussian.
        """
        c, h, w = img.shape[-3:]
        gaussian = torch.normal(mean=0.0, std=self.gaussian_sigma, size=(c, h, w))
        gaussian_image = torch.clamp(img + gaussian, 0.0, 1.0)

        mask = torch.zeros(h, w, dtype=torch.bool)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            # x, y = w//2, h//2

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = True

        patch_gaussian = torch.where(mask == True, gaussian_image, img)

        return patch_gaussian



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def Cutmix(image, target):
    lam = np.random.uniform(0, 1)
    rand_index = torch.randperm(image.size()[0]) #.to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, bbx1:bbx2, bby1:bby2] = target[rand_index, bbx1:bbx2, bby1:bby2]
    return image, target

if __name__ == "__main__":
    testCutmix()
    n_holes, length, gaussian_sigma = 3, 50, 0.5
    img = Image.open('/Users/apple/Desktop/FastFCNresults/nyu-crop.jpg').convert('RGB')
    img = img.resize((480, 480))
    
    # PG
    # input_transform = transform.Compose([transform.ToTensor(), PatchGaussian(n_holes, length, gaussian_sigma)])
    # filename = '%i-%i-%.1f.jpg'%(n_holes, length, gaussian_sigma)
    # Cutout
    input_transform = transform.Compose([transform.ToTensor(), Cutout(n_holes, length)])
    filename = '%i-%i.jpg'%(n_holes, length)
    # Cutmix
    
    img = input_transform(img)
    toPIL = transform.ToPILImage()
    img = toPIL(img)
    img.save('/Users/apple/Desktop/FastFCNresults/'+filename)

