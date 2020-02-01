from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path


def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, id_file, type, transform=None, target_transform=None, loader=default_loader):
        pp='/root/OneDrive/DataSets/CelebA/Img/img_align_celeba/'
        images = []
        targets = []
        id_targets = []
        attr_order = [
            1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35, 7, 19, 27, 29, 30, 34, 6, 14, 16, 22, 21, 24, 36, 37, 38, 0,
            2, 10, 13, 18, 20, 25, 26, 32, 31, 33, 39
        ]
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            images.append(sample[0])
            tmp = [int(i == '1') for i in sample[1:]]
            # tmp = torch.gather(torch.tensor(tmp), 0, attr_order)
            tmp = torch.Tensor(tmp)
            tmp = tmp[attr_order]
            targets.append(tmp)

        for line in open(os.path.join(root, id_file), 'r'):
            sample = line.split()[1]
            id_targets.append(int(sample))

        # self.images = [os.path.join(root, type, img) for img in images]
        self.images = [os.path.join(pp, img) for img in images]
        self.targets = targets
        self.id_targets = id_targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        id_target = self.id_targets[index]
        target = torch.LongTensor(target.long())
        id_target = torch.LongTensor(id_target)
        target = target.long()
        id_target = id_target.long()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)


class TensorSampler(data.Sampler):
    def __init__(self, data_source, batch_size=64):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long().split(self.batch_size))


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class DataLoaderX(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None):
        super().__init__(dataset,
                         batch_size=1,
                         shuffle=False,
                         sampler=None,
                         batch_sampler=None,
                         num_workers=0,
                         collate_fn=None,
                         pin_memory=False,
                         drop_last=False,
                         timeout=0,
                         worker_init_fn=None,
                         multiprocessing_context=None)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())