import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def imread(filename):

    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_batch(file_list, cuda=False):

    images = np.array([imread(str(f)).astype(np.float32) for f in file_list])      
    # Reshape to (n_images, 3, height, width)
    images = images.transpose((0, 3, 1, 2))

    batch = torch.from_numpy(images).type(torch.uint8)
    if cuda:
        batch = batch.cuda()

    return batch


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


def calculate_fid_kid(org_list, output_list, batch_size, cuda):
    """Calculates the FID of two paths"""
    
    print('org length > {}'.format(len(org_list)))
    print('output length > {}'.format(len(output_list)))
    
    
    fid = FrechetInceptionDistance()
    kid = KernelInceptionDistance()
    if cuda:
        fid = fid.cuda()
        kid = kid.cuda()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(org_list), batch_size)):
            start = i
            end = i + batch_size
            org_batch = get_batch(org_list[start:end], cuda)
            fid.update(org_batch, real=True)
            kid.update(org_batch, real=True)
        
        for i in tqdm(range(0, len(output_list), batch_size)):
            start = i
            end = i + batch_size
            output_batch = get_batch(output_list[start:end], cuda)
            fid.update(output_batch, real=False)
            kid.update(output_batch, real=False)
        
    fid_value = fid.compute()
    kid_mean, kid_std = kid.compute()

    return fid_value, kid_mean, kid_std





parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--org', type=str,default='',
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--output', type=str,default='',
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    org_list = get_files_list(args.org,postfix=['*.png'])
    output_list = get_files_list(args.output,postfix=['*.png'])
    
    cuda = False if len(args.gpu)==0 else True
    
    fid, kid_mean, kid_std = calculate_fid_kid(org_list, output_list, batch_size=32, cuda=cuda)

    print('FID: ', fid)
    print('KID_mean: ', kid_mean)
    print('KID_std: ', kid_std)

