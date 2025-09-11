import cv2, os.path as osp, numpy as np, random
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.data.data_util import scandir
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class PairedImageDatasetEdge(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt.get('dataroot_lq', None)
        self.edge_gt_root = opt.get('dataroot_edge_gt', None)
        self.edge_lq_root = opt.get('dataroot_edge_lq', None)
        self.io_backend_opt = opt.get('io_backend', dict(type='disk'))
        self.file_client = None
        # GT 파일 리스트
        self.paths = sorted([osp.join(self.gt_root, p)
                             for p in scandir(self.gt_root, suffix=('.png','.jpg','.jpeg','.bmp'), recursive=True)])
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.scale = opt.get('scale', 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)  # BGR, [0,1]

        # LQ 읽기 or GT에서 축소
        if self.lq_root is not None:
            rel = osp.relpath(gt_path, self.gt_root)
            lq_path = osp.join(self.lq_root, rel)
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)
        else:
            s = self.opt['scale']
            h, w = img_gt.shape[:2]
            img_lq = cv2.resize(img_gt, (w // s, h // s), interpolation=cv2.INTER_CUBIC)

        # HR SAM 엣지 로딩
        edge_gt = None
        if self.edge_gt_root is not None:
            rel_noext = osp.splitext(osp.relpath(gt_path, self.gt_root))[0] + '.npy'
            ep = osp.join(self.edge_gt_root, rel_noext)
            if osp.isfile(ep):
                eg = np.load(ep)
                if eg.ndim == 2: eg = eg[..., None]
                if eg.shape[:2] != img_gt.shape[:2]:
                    eg = cv2.resize(eg, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if eg.ndim == 2: eg = eg[..., None]
                edge_gt = eg.astype(np.float32)

        # LR 엣지 로딩
        edge_lq = None
        if self.edge_lq_root is not None:
            rel_noext = osp.splitext(osp.relpath(gt_path, self.gt_root))[0] + '.npy'
            ep = osp.join(self.edge_lq_root, rel_noext)
            if osp.isfile(ep):
                el = np.load(ep)
                if el.ndim == 2: el = el[..., None]
                if el.shape[:2] != img_lq.shape[:2]:
                    el = cv2.resize(el, (img_lq.shape[1], img_lq.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if el.ndim == 2: el = el[..., None]
                edge_lq = el.astype(np.float32)

        # train augment
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']; scale = self.opt['scale']; lq_ps = gt_size // scale
            h_lq, w_lq = img_lq.shape[:2]
            if h_lq < lq_ps or w_lq < lq_ps:
                raise ValueError(f'LQ size {h_lq}x{w_lq} < patch {lq_ps}')
            top = random.randint(0, h_lq - lq_ps)
            left = random.randint(0, w_lq - lq_ps)
            top_gt, left_gt = top * scale, left * scale

            img_lq = img_lq[top:top+lq_ps, left:left+lq_ps, :]
            img_gt = img_gt[top_gt:top_gt+gt_size, left_gt:left_gt+gt_size, :]
            if edge_gt is not None:
                edge_gt = edge_gt[top_gt:top_gt+gt_size, left_gt:left_gt+gt_size, :]
            if edge_lq is not None:
                edge_lq = edge_lq[top:top+lq_ps, left:left+lq_ps, :]

            to_aug = [img_gt, img_lq]
            idx = {}
            if edge_gt is not None: idx['eg'] = len(to_aug); to_aug.append(edge_gt)
            if edge_lq is not None: idx['el'] = len(to_aug); to_aug.append(edge_lq)
            outs = augment(to_aug, self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq = outs[0], outs[1]
            if 'eg' in idx: edge_gt = outs[idx['eg']]
            if 'el' in idx: edge_lq = outs[idx['el']]
        else:
            # val/test
            h_lq, w_lq = img_lq.shape[:2]; s = self.opt['scale']
            img_gt = img_gt[0:h_lq*s, 0:w_lq*s, :]

        # to tensor
        items = [img_gt, img_lq]; idx = {}
        if edge_gt is not None: idx['eg'] = len(items); items.append(edge_gt)
        if edge_lq is not None: idx['el'] = len(items); items.append(edge_lq)
        tensors = img2tensor(items, bgr2rgb=True, float32=True)
        img_gt_t, img_lq_t = tensors[0], tensors[1]
        edge_gt_t = tensors[idx['eg']] if 'eg' in idx else None
        edge_lq_t = tensors[idx['el']] if 'el' in idx else None

        # normalize (이미지만)
        if self.mean is not None or self.std is not None:
            normalize(img_lq_t, self.mean, self.std, inplace=True)
            normalize(img_gt_t, self.mean, self.std, inplace=True)

        sample = {'lq': img_lq_t, 'gt': img_gt_t, 'lq_path': lq_path, 'gt_path': gt_path}
        if edge_lq_t is not None and self.opt['phase'] == 'train': sample['edge'] = edge_lq_t     # 모델 입력용(LR)
        if edge_gt_t is not None:                                  sample['edge_gt'] = edge_gt_t  # 모니터/손실용(HR)
        return sample
