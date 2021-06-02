from pytracking.tracker.base.basetracker import BaseTracker, SiameseTracker
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from pytracking.tracker.transt.config import cfg
import torchvision.transforms.functional as tvisf
from util.noise import apply_noise


class TransT(SiameseTracker):

    multiobj_mode = 'parallel'

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        # Initialize network
        self.initialize_features()
        # The DiMP network
        self.net = self.params.net
        # Time initialization
        tic = time.time()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0]+bbox[2]/2,
                                    bbox[1]+bbox[3]/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.bbox = bbox

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        self.net.template(z_crop)
        self.net.reset_hidden = True
        out = {'time': time.time() - tic}
        return out

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height


    def track(self, image, info: dict = None, noise=None, noise_mag=None) -> dict:
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))
        x_crop = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        if noise:
            x_crop = apply_noise(x_crop, noise, noise_mag, frame_num=self.frame_num)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)

        # print("Frame: {}. Try resetting hidden state every 8 frames. Quantize prev hidden to [0,1]".format(self.frame_num))
        im_shape = image.shape
        bumps = np.zeros(im_shape[:-1], dtype=np.float32)  # , device=image.device)
        box = [int(x) for x in self.bbox]
        bumps[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = 1
        bumps = bumps[..., None]
        bumps = self.get_subwindow(bumps, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), None)
        bumps = bumps.squeeze()[None, None].float()
        # if self.frame_num % self.net.timesteps == 0:
        #     self.net.force_reset(bumps)

        with torch.no_grad():
            # outputs = self.net.track(x_crop, bumps, info=info)
            outputs = self.net.track(x_crop, bumps, info=info)

        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        # def change(r):
        #     return np.maximum(r, 1. / r)
        #
        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     return np.sqrt((w + pad) * (h + pad))
        # # pred_box:cx,cy,w,h
        # # scale penalty
        # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
        #              (sz(self.size[0]/s_x, self.size[1]/s_x)))
        #
        # # aspect ratio penalty
        # r_c = change((self.size[0]/self.size[1]) /
        #              (pred_bbox[2, :]/pred_bbox[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # pscore = penalty * score

        # window penalty
        pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        # pscore = score
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:,best_idx]

        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x/2
        cy = bbox[1] + self.center_pos[1] - s_x/2

        # smooth bbox
        # no penaty
        width = bbox[2]
        height = bbox[3]


        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        # print(bbox)
        out = {'target_bbox': bbox,
               'best_score': pscore}
        self.frame_num += 1
        return out

