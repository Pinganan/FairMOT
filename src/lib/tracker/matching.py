import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time

from .basetrack import TrackState
from tracker import multitracker

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def appearance_assignment(cost_matrix, length):

    if cost_matrix.size == 0 or not len(cost_matrix) > length:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    lost_matrix = []
    for index, row in enumerate(cost_matrix):
        if index < length:
            continue
        lost_matrix.append(row)
    matched, utrack, udetection = linear_assignment(np.asmatrix(lost_matrix), 0.5)
    for row in matched:
        row[0] += length
    for row in utrack:
        row += length

    return np.asarray(matched), np.asarray(utrack), np.asarray(udetection)


def inf_filter(cost_matrix, unmatched):
    if cost_matrix.size == 0:
        return np.array([]), unmatched
    inf_detection = []
    trans_cost_matrix = cost_matrix.T
    for index, row in enumerate(trans_cost_matrix):
        if row[0] == np.inf and np.all(row == row[0]):
            inf_detection.append(index)
    inf_detection = np.asarray(inf_detection)
    unmatched = np.setdiff1d(unmatched, inf_detection)
    return unmatched, inf_detection


def find_min_assignment(cost_matrix, thresh=1):

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b, temp = [], [], [], []
    row_col = np.where(cost_matrix == np.amin(cost_matrix))
    repeatx, repeaty = -1, -1
    for x, y in zip(row_col[0], row_col[1]):
        if repeatx == x or repeaty == y or cost_matrix[x][y] > thresh:
            continue
        repeatx, repeaty = x, y
        matches.append([x, y])
    matches = np.asarray(matches)
    if matches.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    for x in range(len(cost_matrix)):
        if x not in matches.T[0]:
            unmatched_a.append(x)
    for y in range(len(cost_matrix[0])):
        if y not in matches.T[1]:
            unmatched_b.append(y)
    return matches, np.asarray(unmatched_a), np.asarray(unmatched_b)


def lost_linear_assignment(cost_matrix, matches, u_track, u_detection, length, thresh=2):
    if cost_matrix.size == 0:
        return cost_matrix, np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    for ri in range(cost_matrix.shape[0]):
        cost_matrix[ri<length, ri] = np.inf     # what's the shit
    for x, y in matches:
        cost_matrix[x, :] = np.inf
        cost_matrix[:, y] = np.inf
    m, ut, ud = linear_assignment(cost_matrix, thresh=thresh)
    for row in m:
        matches = np.vstack([matches, row])
    u_track = np.intersect1d(u_track, ut)
    u_detection = np.intersect1d(u_detection, ud)
    return cost_matrix, matches, u_track, u_detection


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def update_appearence(all_features, metric='cosine'):

    feature = np.asarray([feat for feat in all_features], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(feature, feature, metric))

    cost = [sum(i) for i in cost_matrix]

    return all_features[np.argmin(cost)]


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def features_embedding(f1, f2, metric='cosine'):

    cost_matrix = np.zeros((len(f1), len(f2)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(f1, f2, metric))  # Nomalized features
    return cost_matrix


def cal_distance(p1, p2):
    return (((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))**0.5


def distance_standard(cost, standard=24):
    for i in range(len(cost)):
        if cost[i] < standard:
            cost[i] = 0
        else:
            cost[i] = np.inf
    return cost


def short_notice_lost(cost_matrix, tracks, detections, fid, normalize_standard=1):
    if cost_matrix.size == 0:
        return cost_matrix
    cost = []
    detection_xy = [ [d.mapx, d.mapy] for d in detections]
    for t in tracks:
        tracker_xy = [t.mapx, t.mapy]
        cost.append([cal_distance(tracker_xy, d_xy)/((fid-t.frame_id)*normalize_standard) for d_xy in detection_xy])
    return np.asarray(cost)


def tracker_distance(cost_matrix, tracks, detections, fid, avg_standard=24):
    if cost_matrix.size == 0:
        return cost_matrix
    cost = []
    detection_xy = [ [d.mapx, d.mapy] for d in detections]
    for t in tracks:
        tracker_xy = [t.mapx, t.mapy]
        temp = [cal_distance(tracker_xy, d_xy)/(fid-t.frame_id) for d_xy in detection_xy]
        cost.append(distance_standard(temp, standard=avg_standard))
    return np.asarray(cost)


def alltracker_mid_embedding_distance(detections, trackers, metric="cosine"):
    cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)

    cost_matrix = np.empty([0, len(detections)])
    for track in trackers:
        features = np.asarray([f for f in track.features], dtype=np.float)
        matrix = np.maximum(0.0, cdist(det_features, features, metric))
        cost_matrix = np.append(cost_matrix, np.median(matrix, axis=1, keepdims=True).T, axis=0)
    return cost_matrix


def EachDetection_embedding_distance(detections, metric='cosine'):

    cost_matrix = np.zeros((len(detections), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(det_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion_lostStateExcept(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_motion_NoInf(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = 1
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def only_motion(kf, trackers, detections, only_position=False):
    cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(trackers):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row] = gating_distance
    return cost_matrix




