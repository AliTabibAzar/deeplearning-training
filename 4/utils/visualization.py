import torchvision.utils as vutils
import imageio
import os
import glob

def save_image_grid(images, path, nrow=8):
    # denormalize
    images = images * 0.5 + 0.5
    vutils.save_image(images, path, nrow=nrow)

def create_gif(image_folder, output_gif_path, duration=0.5):
    images = []
    filenames = sorted(glob.glob(os.path.join(image_folder, "epoch_*.png")))
    for filename in filenames:
        images.append(imageio.imread(filename))
    
    if len(images) > 0:
        imageio.mimsave(output_gif_path, images, duration=duration)