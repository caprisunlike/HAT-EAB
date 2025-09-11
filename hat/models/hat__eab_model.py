import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import cv2
import numpy as np
import math
from tqdm import tqdm
from os import path as osp

@MODEL_REGISTRY.register()
class HATEABModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.log_dict = {}
        self._l_edge_last = None   # !! add, 마지막 계산된 edge loss 캐시

    # --------------------------------------------
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.edge = data.get('edge', None)
        if self.edge is not None:
            self.edge = self.edge.to(self.device)
        else:
            self.edge = self.canny_from_tensor(self.lq)
        self.edge_gt = data.get('edge_gt', None)
        if self.edge_gt is not None:
            self.edge_gt = self.edge_gt.to(self.device)
        else:
            self.edge_gt = self.canny_from_tensor(self.gt)

        # 디버그 : 3번만 출력
        #if not hasattr(self, '_dbg_edge_seen'):
        #    self._dbg_edge_seen = 0
        #if self._dbg_edge_seen < 3:
        #    print('[DEBUG] edge_gt:',
        #          None if self.edge_gt is None else tuple(self.edge_gt.shape),
        #          'mean=', None if self.edge_gt is None else float(self.edge_gt.mean()))
        #    self._dbg_edge_seen += 1        
    # --------------------------------------------

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

        # edge도 있으면 동일 패딩--------------------------------------------------------------------
        if getattr(self, 'edge', None) is not None:
            self.edge_img = F.pad(self.edge, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
        else:
            self.edge_img = None
        #-----------------------------------------------------------------------------------------

    def process(self):
        # EA 사용 여부 ------------------------------------------------------------------------------------------
        use_branch = self.opt.get('is_train', False) and \
                     self.opt.get('train', {}).get('use_edge_branch', True)
        edge_to_use = self.edge_img if (use_branch and getattr(self, 'edge_img', None) is not None) else None
        # ------------------------------------------------------------------------------------------------------

        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img, edge=edge_to_use)   # !! change : edge=edge_to_use 추가
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img, edge=edge_to_use)   # !! change : edge=edge_to_use 추가
            # self.net_g.train()
        self.edge_band = self.opt.get('train', {}).get('edge_band', 1)   # !! add

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        use_branch = self.opt.get('is_train', False) and self.opt.get('train', {}).get('use_edge_branch', True)   # !! add

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # edge 타일 -------------------------------------------------------------------------------------------------
                edge_tile = None
                if use_branch and getattr(self, 'edge_img', None) is not None:
                    edge_tile = self.edge_img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # ----------------------------------------------------------------------------------------------------------

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile, edge=edge_tile)   # !! change : edge=edge_tile 추가
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile, edge=edge_tile)   # !! change : edge=edge_tile 추가
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        val_edge_sum = None   # !! add
        val_edge_cnt = 0     # !! add

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                #del self.gt

            # ---------------------------------------------------------------------------------
            try:
                #src_name = None
                if getattr(self, 'edge_gt', None) is not None and hasattr(self, 'gt'):
                    edge = self.edge_gt
                    # SR 해상도로 리사이즈
                    if edge.shape[-2:] != self.output.shape[-2:]:
                        edge = F.interpolate(edge, size=self.output.shape[-2:], mode='nearest')
                    # 경계 확장 
                    band = getattr(self, 'edge_band', 1)
                    if band and band > 0:
                        k = 2 * band + 1
                        edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=band)

                    # 계산 확인 : 2개만
                    #if idx < 2:
                    #    s = float(edge.sum().item())
                    #    print(f'[VAL EDGE] src={src_name} sum={s:.1f} shape={tuple(edge.shape)}')

                    if edge.sum().item() > 0:
                        loss_t = self.loss_edge(self.output, self.gt, edge, as_tensor=True)
                        val_edge_sum = loss_t if val_edge_sum is None else (val_edge_sum + loss_t)
                        val_edge_cnt += 1
            except Exception as e:
                self.log_dict['val_edge_err'] = str(e)[:200]
                if idx < 2:
                    print('[VAL EDGE][ERR]', self.log_dict['val_edge_err'])       

            #with torch.no_grad():
            #    s = self.output
            #    g = self.gt if hasattr(self, 'gt') else None
            #    if g is not None:
            #        print(f"[RANGE] SR min/max/mean = {float(s.min()):.4f}/{float(s.max()):.4f}/{float(s.mean()):.4f}")
            #        print(f"[RANGE] GT min/max/mean = {float(g.min()):.4f}/{float(g.max()):.4f}/{float(g.mean()):.4f}")  
            # ---------------------------------------------------------------------------------

            # tentative for out of GPU memory
            if 'gt' in visuals:     # !! add
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # edge loss log ------------------------------------------------------------------------------
        if val_edge_cnt > 0 and val_edge_sum is not None:
            self.log_dict['val_edge_loss'] = float((val_edge_sum / val_edge_cnt).detach())
            if tb_logger is not None:
                try:
                    tb_logger.add_scalar('val/edge_loss', self.log_dict['val_edge_loss'], current_iter)
                except:
                    pass
        else:
            self.log_dict['val_edge_loss'] = float('nan')
        self.log_dict['val_edge_cnt'] = val_edge_cnt

        try:
            import logging
            logging.getLogger('basicsr').info(
                f"VAL edge_loss: {self.log_dict['val_edge_loss']:.6f} (cnt={val_edge_cnt})\n\n"
            )
        except Exception:
            pass
        # --------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------
    # Canny 엣지 맵 생성
    def canny_from_tensor(self, x, low=50, high=200, blur=1):
        """
        x: (B,C,H,W) float tensor. 보통 [0,1] 범위라고 가정.
        반환: (B,1,H,W) float {0,1}
        """
        import torch
        import cv2

        x_f = x.detach()
        if not x_f.is_floating_point():
            x_f = x_f.float()

        # 스케일 조정
        if x_f.max() <= 1.0 + 1e-6:
            x_255 = x_f * 255.0
        else:
            x_255 = x_f

        if x_255.shape[1] == 3:  # 채널 3개인 경우
            w = torch.tensor([0.299, 0.587, 0.114],
                             device=x_255.device, dtype=x_255.dtype).view(1, 3, 1, 1)
            gray255 = (x_255 * w).sum(dim=1)  # (B,H,W), float
        else:  # 채널 1개인 경우
            gray255 = x_255[:, 0, ...]  # (B,H,W)

        # 배치별로 Canny 엣지 추출
        edges = []
        for i in range(gray255.size(0)):
            arr = gray255[i].clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            if blur and blur > 0:
                k = 2 * blur + 1
                arr = cv2.GaussianBlur(arr, (k, k), 0)
            e = cv2.Canny(arr, int(low), int(high))  # (H,W) uint8 {0,255}
            e_t = torch.from_numpy(e).to(x.device, non_blocking=True).float().div(255.0)
            edges.append(e_t.unsqueeze(0))  # (1,H,W)

        return torch.stack(edges, dim=0)  # (B,1,H,W)
    
    def get_current_log(self):
        log = super().get_current_log()
        log['l_edge'] = float('nan')

        cached = getattr(self, '_l_edge_last', None)
        if cached is not None:
            try:
                # 로그 동기화
                log['l_edge'] = cached.detach().cpu().item()
            except Exception:
                log['l_edge'] = float(cached.detach())
            # 재사용 방지
            self._l_edge_last = None
            return log
        return log
    
    def loss_edge(self, sr, gt, edge, *, as_tensor: bool=False):
        """
        sr, gt: (B, C, H, W), w: (B, 1, H, W)  # w는 edge_gt 또는 edge_weight_hr
        """
        if edge.dim() == 3:
            edge = edge.unsqueeze(1)                     # (B,1,H,W)
        edge = edge.to(sr.dtype) 
        abs_err = (gt - sr).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        num = (edge * abs_err).sum()
        den = edge.sum().clamp_min(1e-6)
        out = num / den
        return out if as_tensor else float(out.detach())
    
    def gather_ea_stats(self):
        stats = {'ea_calls': 0, 'ea_skips': 0, 'ea_gate_mean': 0.0, 'ea_out_mean': 0.0, 'ea_edge_mean': 0.0, 'ea_modules': 0}
        for m in self.net_g.modules():
            from hat.archs.hat_eab_arch import EdgeAttention
            if isinstance(m, EdgeAttention):
                stats['ea_modules'] += 1
                stats['ea_calls']   += int(m.dbg_calls)
                stats['ea_skips']   += int(m.dbg_skips)
                stats['ea_gate_mean'] += m.last_gate_mean
                stats['ea_out_mean']  += m.last_out_mean
                stats['ea_edge_mean'] += m.last_edge_mean
        if stats['ea_modules'] > 0:
            for k in ['ea_gate_mean','ea_out_mean','ea_edge_mean']:
                stats[k] /= stats['ea_modules']
        return stats

    def optimize_parameters(self, current_iter):
        # ensure log_dict exists
        if not hasattr(self, 'log_dict') or self.log_dict is None:
            self.log_dict = {}

        # clear old validation keys from training logs
        for k in list(self.log_dict.keys()):
            if k.startswith('val_'):
                self.log_dict.pop(k, None)

        self.optimizer_g.zero_grad()

        edge_branch_on = self.opt.get('train', {}).get('use_edge_branch', True)
        edge_input = getattr(self, 'edge', None) if edge_branch_on else None
        self.output = self.net_g(self.lq, edge=edge_input)

        # pixel loss
        l_pix = None
        if hasattr(self, 'cri_pix') and self.cri_pix is not None:
            l_pix = self.cri_pix(self.output, self.gt)

        # edge loss
        l_edge = None
        if getattr(self, 'edge_gt', None) is not None:
            edge = self.edge_gt
            # SR 출력 해상도에 정렬
            if edge.shape[-2:] != self.output.shape[-2:]:
                edge = F.interpolate(edge, size=self.output.shape[-2:], mode='nearest')
            # 경계 두께 확장
            band = self.opt.get('train', {}).get('edge_band', 1)
            if band and band > 0:
                k = 2 * band + 1
                edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=band)
            l_edge = self.loss_edge(self.output, self.gt, edge, as_tensor=True)
            self._l_edge_last = l_edge.detach() if l_edge is not None else None

        # edge를 학습에 반영할지 결정
        use_edge_loss = bool(self.opt.get('train', {}).get('use_edge_loss', False)) 
        alpha = float(self.opt.get('train', {}).get('alpha', 0.7))

        if use_edge_loss and (l_pix is not None) and (l_edge is not None):   # 픽셀 + 엣지 로스 모두 사용
            # total loss
            total = alpha * l_pix + (1.0 - alpha) * l_edge
            total.backward()
            self.optimizer_g.step()
            # EMA update
            if hasattr(self, 'model_ema'):
                self.model_ema(decay=self.opt.get('train', {}).get('ema_decay', 0.999))
            elif hasattr(self, 'net_g_ema'):
                for p_ema, p in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data * 0.001)
            # 로그
            self.log_dict['l_pix'] = float(l_pix.detach())
            self.log_dict['l_edge'] = float(l_edge.detach())
            self.log_dict['l_total'] = float(total.detach())  # edge 반영 시에만 출력
            #self.log_dict['edge_applied'] = 1
        else:   # 픽셀 로스만 학습 (edge는 로그로만 확인)
            if l_pix is not None:
                l_pix.backward()
            else:
                import torch
                torch.zeros((), device=self.output.device, dtype=self.output.dtype).backward()
            self.optimizer_g.step()
            if hasattr(self, 'model_ema'):
                self.model_ema(decay=self.opt.get('train', {}).get('ema_decay', 0.999))
            elif hasattr(self, 'net_g_ema'):
                for p_ema, p in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data * 0.001)
            # 로그
            if l_pix is not None:
                self.log_dict['l_pix'] = float(l_pix.detach())
            else:
                self.log_dict.pop('l_pix', None)
            if l_edge is not None:
                self.log_dict['l_edge'] = float(l_edge.detach())  # 확인용 출력
            else:
                self.log_dict.pop('l_edge', None)
            self.log_dict.pop('l_total', None)
            #self.log_dict['edge_applied'] = 0

        # EA 디버그 로그
        if self.opt.get('train', {}).get('ea_debug', False):
            ea_stats = self.gather_ea_stats()
            #print('[EA]', ea_stats)
            self.log_dict['ea_gate_m'] = ea_stats['ea_gate_mean']     # 게이트가 평균적으로 열려있는지
            self.log_dict['ea_out_m']  = ea_stats['ea_out_mean']      # EA 출력 평균 크기
            self.log_dict['ea_edge_m'] = ea_stats['ea_edge_mean']     # 입력 edge 평균
            self.log_dict['ea_call']   = ea_stats['ea_calls']         # 누적 호출 횟수
            self.log_dict['ea_skip']   = ea_stats['ea_skips']         # 누적 스킵 횟수
    # ------------------------------------------------------------------------------------------
