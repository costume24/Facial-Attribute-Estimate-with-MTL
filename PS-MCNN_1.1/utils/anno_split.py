import os

filepath = '/media/xuke/SoftWare/BaiduNetdiskDownload/CelebA'
origin = os.path.join(filepath, 'Anno', 'list_attr_celeba.txt')
ids = os.path.join(filepath, 'Anno', 'identity_CelebA.txt')

TRAIN_STOP = 162770
VALID_STOP = 182637

with open(origin, 'r') as f:
    all = f.readlines()[2:]
    path = os.path.join(filepath, 'Anno', 'list_attr_celeba_train.txt')
    with open(path, 'a') as g:
        g.writelines(all[:TRAIN_STOP])
    path = os.path.join(filepath, 'Anno', 'list_attr_celeba_val.txt')
    with open(path, 'a') as g:
        g.writelines(all[TRAIN_STOP:VALID_STOP])
    path = os.path.join(filepath, 'Anno', 'list_attr_celeba_test.txt')
    with open(path, 'a') as g:
        g.writelines(all[VALID_STOP:])
    print('[*] Attr anno split success!')

with open(ids, 'r') as f:
    all = f.readlines()
    path = os.path.join(filepath, 'Anno', 'identity_CelebA_train.txt')
    with open(path, 'a') as g:
        g.writelines(all[:TRAIN_STOP])
    path = os.path.join(filepath, 'Anno', 'identity_CelebA_val.txt')
    with open(path, 'a') as g:
        g.writelines(all[TRAIN_STOP:VALID_STOP])
    path = os.path.join(filepath, 'Anno', 'identity_CelebA_test.txt')
    with open(path, 'a') as g:
        g.writelines(all[VALID_STOP:])
    print('[*] ID anno split success!')
