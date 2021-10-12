from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import time

from map import MapTable
from tracker import matching
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq_multiLoader(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    table = MapTable()
    image_map = cv2.imread("MAP.jpg")
    dataloader_amount = len(dataloader)
    timer = Timer()
    results = []
    frame_id = 0
    tracker = []
    port_list = [221, 225]  #pm port
    for i in range(dataloader_amount):
        tracker.append(JDETracker(opt, frame_rate=frame_rate, port = port_list[i]))

    for frame_counter in range(len(dataloader[0])):
        '''
        if frame_counter % 4 != 0:
            for dataloader_index in range(dataloader_amount):
                dataloader[dataloader_index].__next__()
            continue
        '''
        if frame_counter < 36:
            for dataloader_index in range(dataloader_amount):
                dataloader[dataloader_index].__next__()
            continue

        #if frame_id % 20 == 0:
            #logger.debug('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        timer.tic()

        # get detections
        images = []
        detections = []
        for dataloader_index in range(dataloader_amount):
            (path, img, img0) = dataloader[dataloader_index].__next__()
            images.append(img0)
            if use_cuda:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            detections.append(tracker[dataloader_index].get_detection(blob, img0))

        # run tracking
        online_targets = []
        online_tlwhs = []
        online_ids = []
        for dataloader_index in range(dataloader_amount):
            # detections_xy to map_xy
            tracker[dataloader_index].map_detection(detections[dataloader_index])
            # update
            online_targets.append(tracker[dataloader_index].update(detections[dataloader_index]))

        for i in online_targets:
            tlwhs = []
            ids = []
            for t in i:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    tlwhs.append(tlwh)
                    ids.append(tid)
            online_tlwhs.append(tlwhs)
            online_ids.append(ids)

        map_tlwhs = []
        map_ids = []
        # match
        two_matches, two_ids, matches_a, id_a, matches_b, id_b, u1, u2 = table.search_match(online_targets)

        for (i, j), m_id in zip(two_matches, two_ids):
            t1 = online_targets[0][i]
            t2 = online_targets[1][j]
            dot = ((t1.mapx+t2.mapx)/2, (t1.mapy+t2.mapy)/2)
            map_tlwhs.append(dot)
            map_ids.append(m_id)
        for a, a_id in zip(matches_a, id_a):
            t1 = online_targets[0][a]
            map_tlwhs.append((t1.mapx, t1.mapy))
            map_ids.append(a_id)
        for b, b_id in zip(matches_b, id_b):
            t1 = online_targets[1][b]
            map_tlwhs.append((t1.mapx, t1.mapy))
            map_ids.append(b_id)

        # single_match
        online_targets[0] = [online_targets[0][i] for i in u1]
        online_targets[1] = [online_targets[1][i] for i in u2]
        two_matches, two_ids, matches_a, id_a, matches_b, id_b = table.search_single(online_targets)

        for (i, j), m_id in zip(two_matches, two_ids):
            t1 = online_targets[0][i]
            t2 = online_targets[1][j]
            dot = ((t1.mapx+t2.mapx)/2, (t1.mapy+t2.mapy)/2)
            map_tlwhs.append(dot)
            map_ids.append(m_id)
        for a, a_id in zip(matches_a, id_a):
            t1 = online_targets[0][a]
            map_tlwhs.append((t1.mapx, t1.mapy))
            map_ids.append(a_id)
        for b, b_id in zip(matches_b, id_b):
            t1 = online_targets[1][b]
            map_tlwhs.append((t1.mapx, t1.mapy))
            map_ids.append(b_id)

        # draw id table
        if table.match or table.single_a or table.single_b:
            print()
            print("    ID    C1    C2")
            print("------------------")
        for i, ai, bi in zip(table.match, table.match_a, table.match_b):
            print("{:6d}{:6d}{:6d}".format(i, ai, bi))
        for i, ai in table.single_a.items():
            print("{:6d}{:6d}".format(i, ai))
        for i, bi in table.single_b.items():
            print("{:6d}{:12d}".format(i, bi))

        timer.toc()
        # save results
        for i in range(dataloader_amount):
            temp = []
            for tlwh, ids in zip(online_tlwhs[i], online_ids[i]):
                temp.append((frame_id + 1, tlwh, ids))
            results.append(temp)

        online_ims = []
        if show_image or save_dir is not None:
            for i in range(dataloader_amount):
                online_im = vis.plot_tracking(images[i], online_tlwhs[i], online_ids[i], frame_id=frame_id,
                                                fps=1. / timer.average_time)
                #online_im = images[i]
                online_ims.append(online_im)
            online_map = vis.plot_pixel(image_map, map_tlwhs, map_ids)
            image_map = online_map
        if show_image:
            for i in range(dataloader_amount):
                cv2.imshow('online_im' + str(i), online_ims[i])
            cv2.imshow('map_im', image_map)
            if cv2.waitKey(1) == ord('q'):
                break
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir + "/m",  '{:d}.jpg'.format(frame_id)), image_map)
            cv2.imwrite(os.path.join(save_dir + "/c1", '{:d}.jpg'.format(frame_id)), online_ims[0])
            cv2.imwrite(os.path.join(save_dir + "/c2", '{:d}.jpg'.format(frame_id)), online_ims[1])
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls
            

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0

    # for checking model dtection ability
    frame_mark = []
    detectionNum = []
    detectionSet = []

    for i, (path, img, img0) in enumerate(dataloader):

        if i % 4 != 0:
            continue
        if frame_id % 20 == 0:
            logger.debug('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        
        '''
        # record frame_id, feature, and amount of detection
        if len(tracker.get_detection(blob, img0)) > 2:
            if len(frame_mark) == 0:
                frame_mark.append(frame_id)
                detectionSet += tracker.get_detection(blob, img0)
                detectionNum.append(len(detectionSet))
            elif frame_id-20 > frame_mark[-1]:
                frame_mark.append(frame_id)
                detectionSet += tracker.get_detection(blob, img0)
                detectionNum.append(len(detectionSet))
        '''

        online_targets = tracker.update(tracker.get_detection(blob, img0))
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            #print(online_tlwhs, online_ids, frame_id)
        if show_image:
            cv2.imshow('online_im', online_im)
            if cv2.waitKey(1) == ord('q'):
                break
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)

    '''
    cost_matrix = matching.EachDetection_embedding_distance(detectionSet)
    ccc = 0
    for inde, (de, ro) in enumerate(zip(detectionSet, cost_matrix)):

        print("{} {}  ".format(de.tlwh[:2], [int(a*100) for a in ro]))

        if (inde+1) in detectionNum:
            print("--"*60 + " " + str(frame_mark[ccc]))
            ccc += 1
            print()
    '''


    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=True, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
