import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F
import time

from models.model import create_model, load_model
from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.features.append(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = matching.update_appearence(self.features)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated= False
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.det_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def get_detection(self, im_blob, img0):
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []
        print()
        return detections

    def update(self, detections):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        unconfirmed_stracks = []
        
        ''' step1 diff tracker(u/s/l) '''
        unconfirmed = []
        strack_pool = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                strack_pool.append(track)
        strack_len = len(joint_stracks(strack_pool, self.lost_stracks))
        print("1st detection amount " + str(len(detections)))
        print("    allstrack amount " + str(strack_len))
        print("    unchecked amount " + str(len(unconfirmed)))
        print("    lostState amount " + str(len(self.lost_stracks)))
        print()


        ''' step2-1 embedding for appearance '''
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #just_terminal_display(dists, strack_pool, "embedding_distance")


        ''' step2-2 fuse_motion for location in future '''
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.3)
        u_detection, inf_detection = matching.inf_filter(dists, u_detection)   # for inf value
        just_terminal_display(dists, strack_pool, "s_tracker & all_detection fuse_motion")
        print("2nd matrixs amount " + str(dists.shape))
        print("    matches amount " + str(len(matches)))
        print("    tracker amount " + str(len(u_track)))
        print("    nor_det amount " + str(len(u_detection)))
        print("    inf_det amount " + str(len(inf_detection)))
        print()
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        '''
            if detection was valuable with tracker, it possibly matches trackers that means detection belongs to STrack
            instead, it doesn't matches anyone that means detections are classified as Unconfirmed Tracker
        '''
        inf_detections = [detections[i] for i in inf_detection]      # => inf detect, it doesn't match any tracker
        val_detections = [detections[i] for i in u_detection]        # => valuable detect, maybe match tracker


        ''' step3 iou '''
        strack_pool = [strack_pool[i] for i in u_track]
        dists = matching.iou_distance(strack_pool, val_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.2)
        just_terminal_display(dists, strack_pool, "tracker & nor_detection iou_distance")
        print("3rd matrixs amount " + str(dists.shape))
        print("    matches amount " + str(len(matches)))
        print("    tracker amount " + str(len(u_track)))
        print("    nor_det amount " + str(len(u_detection)))
        print()

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = val_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, False)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        ''' step4 lost embedding for appearance '''
        strack_pool = self.lost_stracks  
        # matching.frame_distance(strack_pool, inf_detections, self.frame_id)
        dists = matching.embedding_distance(strack_pool, inf_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        just_terminal_display(dists, strack_pool, "lost_tracker & inf_detection embedding")
        print("4th matrixs amount " + str(dists.shape))
        print("    matches amount " + str(len(matches)))
        print("    l_track amount " + str(len(u_track)))
        print("    inf_det amount " + str(len(u_detection)))
        print()

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = inf_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        ''' step5 tracker(u) -> stracker(s) '''
        inf_detections = [inf_detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, inf_detections)
        matches, u_unconfirmed, u_detection = matching.find_min_assignment(dists)
        just_terminal_display(dists, unconfirmed, "u_tracker & inf_detection iou_distance")
        print("5th matrixs amount " + str(dists.shape))
        print("    matches amount " + str(len(matches)))
        print("    u_track amount " + str(len(u_unconfirmed)))
        print("    inf_det amount " + str(len(u_detection)))
        print()

        for itracked, idet in matches:
            unconfirmed[itracked].update(inf_detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        ''' step6 detection -> tracker(u) '''
        allow_number = 0
        score_list = {}
        for inew in u_detection:
            track = inf_detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            if track.track_id > 8:
                break
            allow_number = allow_number + 1
            score_list[track.track_id] = track.score
            unconfirmed_stracks.append(track)
        print("6th isVaild amount " + str(allow_number))
        print(score_list)
        print()

        """ don't care """
        '''
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        '''

        # print('Ramained match {} s'.format(t4-t3))
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, unconfirmed_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        #self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        logger.debug('========Frame {}======='.format(self.frame_id))
        logger.debug('Unconfirmed: {}'.format([track.track_id for track in unconfirmed_stracks]))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Unconfirmed: {}'.format([track.track_id for track in unconfirmed_stracks]))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('AllLost: {}'.format([track.track_id for track in self.lost_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def just_terminal_display(matrix, stracks, matrix_mark = "IOU"):

    if matrix.size == 0:
        return
    print(matrix_mark + " matrix :")
    for i, m in zip(stracks, matrix):
        print(i.track_id, m)
    print()
