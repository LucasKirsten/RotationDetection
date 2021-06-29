import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def _adjust_boxes(boxes, h, w):
    boxes[:,:,0] *= w
    boxes[:,:,1] *= h
    boxes = np.int0(boxes)
    return boxes

def _get_pts(path_ann):
    ann = open(path_ann, 'r').read()
    ann = ann.split('\n')
    
    anns = []
    for box in ann:
        an = list(map(float, box.split(',')[1:-2]))
        if len(an)>1:
            anns.append(np.array([
                [an[0],an[4]],
                [an[1],an[5]],
                [an[2],an[6]],
                [an[3],an[7]]]))
    return np.array(anns)

def _get_ann(path_annotations):
    
    annotations = {}
    for path in path_annotations:
        boxes = _get_pts(path)
        annotations[path] = boxes
        
    return annotations

def convert2dota(path_imgs, path_ann, path_save):
    
    anns = _get_ann(path_ann)
    for i in tqdm(range(len(path_imgs))):
        path_img = path_imgs[i]
        ann = anns[path_ann[i]]
        
        img = cv2.imread(path_img)
        h,w,_ = img.shape
        
        boxes = _adjust_boxes(np.copy(ann), h, w)
        
        label_name = os.path.split(path_ann[i])[-1]
        
        with open(os.path.join(path_save, label_name), 'w') as file:
            file.write('imagesource:UFRGS\n')
            file.write('gsd:0\n')
            for box in boxes:
                box = map(str, box.reshape(-1))
                box = ' '.join(list(box))
                file.write(box + ' cell 0\n')
    
if __name__=='__main__':
    
    # change here
    path_img = './dataset/test/imgs'
    path_ann = './dataset/test/annotations/alpr_format'
    path_save = './dataset/test/annotations/dota_format'
    
    os.makedirs(path_save, exist_ok=True)
    path_imgs = sorted(glob(os.path.join(path_img, '*')))
    path_ann  = sorted(glob(os.path.join(path_ann, '*')))
    convert2dota(path_imgs, path_ann, path_save)