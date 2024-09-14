# Copyright (c) OpenMMLab. All rights reserved.
import time
from multiprocessing import Pool

import motmetrics as mm
import numpy as np
import pandas as pd
from mmcv.utils import print_log
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from motmetrics.lap import linear_sum_assignment
from motmetrics.math_util import quiet_divide
import torch


METRIC_MAPS = {
    'idf1': 'IDF1',
    'mota': 'MOTA',
    'motp': 'MOTP',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDSw',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_fragmentations': 'FM'
}


def bbox_distances(bboxes1, bboxes2, iou_thr=0.5):
    """Calculate the IoU distances of two sets of boxes."""
    ious = bbox_overlaps(bboxes1, bboxes2, mode='iou')
    distances = 1 - ious
    distances = np.where(distances > iou_thr, np.nan, distances)
    return distances


def outs2results(bboxes=None,
                 labels=None,
                 ids=None,
                 num_classes=None,
                 **kwargs):
    assert labels is not None
    assert num_classes is not None

    results = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        [bboxes[labels == i, :], ids[labels == i, None]],
                        axis=1) for i in range(num_classes)
                ]
        else:
            raise NotImplementedError
        results['bbox_results'] = bbox_results

    return results


def acc_single_video(results,
                     gts,
                     iou_thr=0.5,
                     ignore_iof_thr=0.5,
                     ignore_by_classes=False,
                     in_tracking_class=None):
    """Accumulate results in a single video."""
    num_classes = len(results[0])
    accumulators = [
        mm.MOTAccumulator(auto_id=True) for i in range(num_classes)
    ]
    for result, gt in zip(results, gts):
        # if ignore_by_classes:
        #     gt_ignore = outs2results(
        #         bboxes=gt['bboxes_ignore'],
        #         labels=gt['labels_ignore'],
        #         num_classes=num_classes)['bbox_results']
        # else:
        #     gt_ignore = [gt['bboxes_ignore'] for i in range(num_classes)]
        gt_ignore = [[] for _ in range(num_classes)]
        gt = outs2results(
            bboxes=gt['bboxes'],
            labels=gt['labels'],
            ids=gt['instance_inds'],
            num_classes=num_classes)['bbox_results']
        for i in range(num_classes):
            if in_tracking_class is not None:
                if not in_tracking_class[i]:
                    continue
            gt_ids, gt_bboxes = gt[i][:, -1].astype(np.int), gt[i][:, :4]
            pred_ids, pred_bboxes = result[i][:, 5].astype(np.int), result[i][:, :4]
            dist = bbox_distances(gt_bboxes, pred_bboxes, iou_thr)
            if len(gt_ignore[i]) > 0:
                # 1. assign gt and preds
                fps = np.ones(pred_bboxes.shape[0]).astype(np.bool)
                row, col = linear_sum_assignment(dist)
                for m, n in zip(row, col):
                    if not np.isfinite(dist[m, n]):
                        continue
                    fps[n] = False
                # 2. ignore by iof
                iofs = bbox_overlaps(pred_bboxes, gt_ignore[i], mode='iof')
                ignores = (iofs > ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = ~(fps & ignores)
                pred_ids = pred_ids[valid_inds]
                dist = dist[:, valid_inds]
            if dist.shape != (0, 0):
                accumulators[i].update(gt_ids, pred_ids, dist)
    return accumulators


def aggregate_accs(accumulators, classes):
    """Aggregate results from each class."""
    # accs for each class
    items = list(classes)
    names, accs = [[] for c in classes], [[] for c in classes]
    for video_ind, _accs in enumerate(accumulators):
        for cls_ind, acc in enumerate(_accs):
            if len(acc._events['Type']) == 0:
                continue
            name = f'{classes[cls_ind]}_{video_ind}'
            names[cls_ind].append(name)
            accs[cls_ind].append(acc)

    # overall
    items.append('OVERALL')
    names.append([n for name in names for n in name])
    accs.append([a for acc in accs for a in acc])

    return names, accs, items


def eval_single_class(names, accs):
    """Evaluate CLEAR MOT results for each class."""
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=METRIC_MAPS.keys(), generate_overall=True)
    results = [v['OVERALL'] for k, v in summary.to_dict().items()]
    motp_ind = list(METRIC_MAPS).index('motp')
    if np.isnan(results[motp_ind]):
        num_dets = mh.compute_many(
            accs,
            names=names,
            metrics=['num_detections'],
            generate_overall=True)
        sum_motp = (summary['motp'] * num_dets['num_detections']).sum()
        motp = quiet_divide(sum_motp, num_dets['num_detections']['OVERALL'])
        results[motp_ind] = float(1 - motp)
    else:
        results[motp_ind] = 1 - results[motp_ind]
    return results


def eval_mot(results,
             annotations,
             logger=None,
             classes=None,
             iou_thr=0.5,
             ignore_iof_thr=0.5,
             ignore_by_classes=False,
             in_tracking_class=None,
             nproc=4):
    """Evaluation CLEAR MOT metrics.

    Args:
        results (list[list[list[ndarray]]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            categories. The ndarray indicates the tracking results.
        annotations (list[list[dict]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            the annotations of each video. Keys of annotations are

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `instance_ids`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        logger (logging.Logger | str | None, optional): The way to print the
            evaluation results. Defaults to None.
        classes (list, optional): Classes in the dataset. Defaults to None.
        iou_thr (float, optional): IoU threshold for evaluation.
            Defaults to 0.5.
        ignore_iof_thr (float, optional): Iof threshold to ignore results.
            Defaults to 0.5.
        ignore_by_classes (bool, optional): Whether ignore the results by
            classes or not. Defaults to False.
        nproc (int, optional): Number of the processes. Defaults to 4.

    Returns:
        dict[str, float]: Evaluation results.
    """
    print_log('---CLEAR MOT Evaluation---', logger)
    t = time.time()
    gts = annotations.copy()
    if classes is None:
        classes = [i + 1 for i in range(len(results[0]))]
    assert len(results) == len(gts)
    metrics = METRIC_MAPS.keys()

    print_log('Accumulating...', logger)

    pool = Pool(nproc)
    accs = pool.starmap(
        acc_single_video,
        zip(results, gts, [iou_thr for _ in range(len(gts))],
            [ignore_iof_thr for _ in range(len(gts))],
            [ignore_by_classes for _ in range(len(gts))], [in_tracking_class for _ in range(len(gts))]))
    names, accs, items = aggregate_accs(accs, classes)
    print_log('Evaluating...', logger)
    eval_results = pd.DataFrame(columns=metrics)
    summaries = pool.starmap(eval_single_class, zip(names, accs))
    pool.close()

    # category and overall results
    for i, item in enumerate(items):
        eval_results.loc[item] = summaries[i]

    dtypes = {m: type(d) for m, d in zip(metrics, summaries[0])}
    # average results
    avg_results = []
    for i, m in enumerate(metrics):
        v = np.array([s[i] for s in summaries[:len(classes)]])
        v = np.nan_to_num(v, nan=0)
        if dtypes[m] == int:
            avg_results.append(int(v.sum()))
        elif dtypes[m] == float:
            avg_results.append(float(v.mean()))
        else:
            raise TypeError()
    eval_results.loc['AVERAGE'] = avg_results
    eval_results = eval_results.astype(dtypes)

    print_log('Rendering...', logger)
    strsummary = mm.io.render_summary(
        eval_results,
        formatters=mm.metrics.create().formatters,
        namemap=METRIC_MAPS)

    print_log('\n' + strsummary, logger)
    print_log(f'Evaluation finishes with {(time.time() - t):.2f} s.', logger)

    eval_results = eval_results.to_dict()
    out = {METRIC_MAPS[k]: v['OVERALL'] for k, v in eval_results.items()}
    for k, v in out.items():
        out[k] = float(f'{(v):.3f}') if isinstance(v, float) else int(f'{v}')
    for m in ['OVERALL', 'AVERAGE']:
        out[f'track_{m}_copypaste'] = ''
        for k in METRIC_MAPS.keys():
            v = eval_results[k][m]
            v = f'{(v):.3f} ' if isinstance(v, float) else f'{v} '
            out[f'track_{m}_copypaste'] += v

    return out
