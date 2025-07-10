#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
import agent
from utils import PIDController
import time

FRAME_CX = 320 - 39
FRAME_CY = 240 - 10

def overlay_masks(frame, mask, alpha=0.2):
    mask_3c = (mask[..., None] > 0.5).astype(np.uint8) * np.asarray((255,0,255), np.uint8)
    return cv2.addWeighted(frame, 1.0, mask_3c, alpha, 0.0)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap=5):
    """在两点间绘制虚线。"""
    # 计算两点之间的距离
    dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist == 0:
        return
    # 单位向量
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    step = dash_length + gap
    for i in range(0, dist, step):
        start_x = int(pt1[0] + dx * i)
        start_y = int(pt1[1] + dy * i)
        end_x = int(pt1[0] + dx * min(i + dash_length, dist))
        end_y = int(pt1[1] + dy * min(i + dash_length, dist))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def get_target_centroids(frame, mask, conf=None, radius=3):
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None, None
    cx = int(xs.mean())
    cy = int(ys.mean())

    # 绘制掩码中心点
    cv2.circle(frame, (cx, cy), radius, (255, 255, 0), -1)

    # 绘制图像中心点
    center_pt = (FRAME_CX, FRAME_CY)
    cv2.circle(frame, center_pt, radius, (0, 255, 0), -1)

    # 绘制虚线连接图像中心与掩码中心
    draw_dashed_line(frame, center_pt, (cx, cy), (0, 255, 255), thickness=1, dash_length=10, gap=6)

    # 绘制置信度文本（可选）
    if conf is not None:
        text = f"{conf:.2f}"
        cv2.putText(frame, text, (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return cx, cy

def get_best_mask(results, frame_shape, conf_thresh: float = 0.75):
    """根据 YOLO segmentation 结果提取并处理掩码。

    1. 根据置信度阈值筛选掩码（> conf_thresh）。
    2. 裁剪掉 letterbox 填充，使掩码尺寸与帧一致。
    3. 如果存在多个掩码，将它们合并为单一掩码，并取最高置信度。

    返回 (mask_proc, conf_proc)。若无满足条件的掩码，返回 (None, None)。
    """
    if results.masks is None:
        return None, None

    # 原始掩码 (N, H_pad, W_pad)
    raw_masks = results.masks.data.cpu().numpy()

    # 置信度
    confs = None
    if results.boxes is not None and results.boxes.conf is not None:
        confs = results.boxes.conf.cpu().numpy()

    # 置信度过滤
    if confs is not None:
        keep_idx = confs > conf_thresh
        confs = confs[keep_idx]
        raw_masks = raw_masks[keep_idx]

    if len(raw_masks) == 0:
        return None, None

    # 去除 letterbox 填充，保证尺寸一致
    frame_h, frame_w = frame_shape[:2]
    mask_h, mask_w = raw_masks.shape[1:]
    pad_y = max((mask_h - frame_h) // 2, 0)
    pad_x = max((mask_w - frame_w) // 2, 0)

    masks = np.array([
        m[pad_y:pad_y + frame_h, pad_x:pad_x + frame_w]
        for m in raw_masks
    ])

    if len(masks) > 1:
        # 合并掩码
        mask_proc = (np.sum(masks, axis=0) > 0).astype(np.uint8)
        conf_proc = float(confs.max()) if confs is not None else None
    else:
        mask_proc = masks[0]
        conf_proc = float(confs[0]) if confs is not None else None

    return mask_proc, conf_proc

class Detector:
    """封装相机读取、模型推理、结果渲染的完整流程。"""

    def __init__(self, model_path: str = "./weights/best_v3.engine", conf_thresh: float = 0.75):
        self.model = YOLO(model_path, task="segment")
        self.conf_thresh = conf_thresh

        self.px = 0
        self.py = 0
        self.dpx = 0
        self.dpy = 0
        self.missing = True

        self.pid_far_x = PIDController(Kp=6, Ki=1, Kd=7, output_limits=(-1500, 1500))
        self.pid_far_y = PIDController(Kp=2, Ki=0.0, Kd=1, output_limits=(-1500, 1500))
        self.pid_near_x = PIDController(Kp=4, Ki=0.8, Kd=4, output_limits=(-1500, 1500))
        self.pid_near_y = PIDController(Kp=1.5, Ki=0.1, Kd=0.1, output_limits=(-1500, 1500))

    def track_point(self, frame, dist):
        """对单帧进行推理与可视化处理，返回处理后帧以及 (cx, cy)。"""
        results = self.model(frame, verbose=False)[0]
        
        mask_proc, conf_proc = get_best_mask(results, frame.shape, conf_thresh=self.conf_thresh)
        dpx = 0
        dpy = 0
        if mask_proc is not None:
            self.missing = False
            frame = overlay_masks(frame, mask_proc, alpha=0.2)
            self.px, self.py = get_target_centroids(frame, mask_proc, conf_proc)
            if self.px is not None and self.py is not None:
                dpx = self.px - FRAME_CX
                dpy = self.py - FRAME_CY
        else:
            self.missing = True
            print("missing,reset pid")
            self.pid_far_x.reset()
            self.pid_far_y.reset()
            self.pid_near_x.reset()
            self.pid_near_y.reset()

        if dist > 250:
            self.dpx = int(self.pid_far_x.update(dpx))
            self.dpy = int(self.pid_far_y.update(dpy))
        else:
            self.dpx = int(self.pid_near_x.update(dpx))
            self.dpy = int(self.pid_near_y.update(dpy))

        cv2.putText(frame, f"dpx: {self.dpx}, dpy: {self.dpy}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        
        return frame

def main():
    detector = Detector()
    client = agent.EthClient()

    print("按 'q' 键退出。")
    while True:
        frame = client.rcv_frame()
        if frame is None:
            continue

        frame = detector.track_point(frame, 300)

        cv2.imshow("YOLOv11-Seg 实时分割", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    client.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
