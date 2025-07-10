from torchvision import transforms

def transform_settings(aug= False):
    if aug == False:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)) # Normalize images, converge faster and more stable
            ])
        return transform
    else:
        transform_augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding= 4), # Square crop of size x size, pad all border
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly change the brightness
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])     
        return transform_augmentation