import cv2
from imgaug import augmenters as iaa
import os
import argparse


def parse_args():
    parse = argparse.ArgumentParser(description='Transform dataset from clear to snowy')
    parse.add_argument('-i', '--input', type=str, help='Directory of input images')
    parse.add_argument('-o', '--output', type=str, help='Directory of output images')
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    seq = iaa.Sequential([
    iaa.imgcorruptlike.Fog(severity=1),
    iaa.imgcorruptlike.Snow(severity=1),
    ])

    path = args.input
    savepath = args.output

    imglist = []
    filelist = os.listdir(path)

    for item in filelist:
        img = cv2.imread(os.path.join(path, item))
        imglist.append(img)

    print('Load Successfully!')

    total_num = len(imglist)
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for index in range(total_num):
        img_aug = seq.augment_image(imglist[index])
        filename = str(filelist[index])
        cv2.imwrite(os.path.join(savepath, filename), img_aug)
        print(f'Image {index + 1} / {total_num} has been saved.', end='\r')
    
    print()
    print('Done!')


if __name__ == '__main__':
    main()
