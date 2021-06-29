import os
import cv2
from glob import glob
from tqdm import tqdm

path_root = '/workdir/datasets/msc/UFRGS_CELL_2classes'

path_images = glob(os.path.join(path_root, 'train', 'imgs', '*'))
path_images.extend( glob(os.path.join(path_root, 'test', 'imgs', '*')) )

for path in tqdm(path_images):
    
    if path.endswith('.jpg'):
        continue
    
    img = cv2.imread(path)
    ext = path.split('.')[-1]
    cv2.imwrite(path.replace(ext, 'jpg'), img)