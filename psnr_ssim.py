import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import numpy
import argparse


parser = argparse.ArgumentParser(description="training configs")
parser.add_argument("--org", type=str, default="", help="starting epoch")
parser.add_argument("--output", type=str, default="", help="epoch")
args = parser.parse_args()


def getFilePathList(file_dir):
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix=None):
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix is None:
        file_list = filePath_list
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


def compute_psnr(imfil1,imfil2):
    img1=cv2.imread(imfil1)
    img2=cv2.imread(imfil2)
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return peak_signal_noise_ratio(img1,img2)

def compute_ssim(imfil1,imfil2):
    img1=cv2.imread(imfil1)
    img2=cv2.imread(imfil2)
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(img1,img2)


def psnr():
    psnr_values = []
    file_list = get_files_list(args.output,postfix=['*.png'])   
    for file_path in tqdm(file_list):
        ref_file = os.path.join(args.org,file_path.split('/')[-1])
        psnr_values.append(compute_psnr(ref_file, file_path) )
    return numpy.mean(psnr_values)

def ssim():
    ssim_values = []
    file_list = get_files_list(args.output,postfix=['*.png'])   
    for file_path in tqdm(file_list):
        ref_file = os.path.join(args.org,file_path.split('/')[-1])
        ssim_values.append( compute_ssim(ref_file, file_path) )
    return numpy.mean(ssim_values)



if __name__ == '__main__':
    args = parser.parse_args()

    psnr_value = psnr()
    ssim_value = ssim()

    print('PSNR: %.2f \nSSIM: %.2f' % (psnr_value, ssim_value))