from torchvision import transforms

bright_t = transforms.ColorJitter(brightness = [1,3])
contrast_t = transforms.ColorJitter(contrast = [2,6])
saturation_t = transforms.ColorJitter(saturation = [1,4])
hue_t = transforms.ColorJitter(hue = 0.3)
gs_t = transforms.Grayscale(3)
hflip_t = transforms.RandomHorizontalFlip(p = 1)
rp_t = transforms.RandomPerspective(p = 1, distortion_scale = 0.5)
rot_t = transforms.RandomRotation(degrees = 90)
blur_t = transforms.GaussianBlur(kernel_size=7, sigma=(0.3, 0.7))
vflip_t = transforms.RandomVerticalFlip(p = 1)
sol_t = transforms.RandomSolarize(p = 1, threshold = 0.4)

aug_transformations = {
    "CS-HF": transforms.Compose([contrast_t, saturation_t, hflip_t]),
    "H-RP": transforms.Compose([hue_t, rp_t]),
    "B-GS-R": transforms.Compose([bright_t, gs_t, rot_t]),
    "S-BL-VF": transforms.Compose([sol_t, blur_t, vflip_t])
}