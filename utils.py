#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import zmq
import time
import threading
import queue
import numpy as np
import math
import bisect

MINE_Y = 300
MINE1_X = 150
MINE2_X = 300
MINE3_X = 450
MINE1_COORD = (MINE1_X, MINE_Y)
MINE2_COORD = (MINE2_X, MINE_Y)
MINE3_COORD = (MINE3_X, MINE_Y)
THRESHOLDS = [0, 225, 375, 600]
RESULT_DEST = [150, 300, 450]
MINE_RADIUS = 15
MINE_DANGER_RADIUS = 75
MINE_SIDE = 75

class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, *, setpoint=0.0, output_limits=(None, None)):
        import time
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._time = time  # 本地缓存，避免频繁全局查找

    def reset(self):
        """重置内部状态。"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, measurement):
        """根据测量值计算 PID 输出。"""
        now = self._time.time()
        error = self.setpoint - measurement

        # 积分项
        self._integral += error

        # 微分项
        derivative = error - self._prev_error

        # PID 输出
        output = (
            self.Kp * error +
            self.Ki * self._integral +
            self.Kd * derivative
        )

        # 输出限幅
        low, high = self.output_limits
        if low is not None and output < low:
            output = low
        elif high is not None and output > high:
            output = high

        # 更新存储的状态
        self._prev_error = error
        self._prev_time = now
        return output 

def draw_map() -> np.ndarray:
    """
    生成静态赛场底图（600×600）。
    """
    _map = np.zeros((600, 600, 3), np.uint8)

    # 起终点、边框、分界线
    cv2.rectangle(_map, (0, 0), (89, 59), (128, 32, 32), -1)        # 起点
    cv2.rectangle(_map, (510, 540), (599, 599), (128, 32, 32), -1)  # 终点
    cv2.line(_map, (0, 299), (599, 299), (255, 255, 255), 3)        # 中线
    cv2.rectangle(_map, (0, 0), (598, 598), (255, 255, 255), 3)     # 边框

    # 三个雷区
    for center in (MINE1_COORD, MINE2_COORD, MINE3_COORD):
        cv2.circle(_map, center, radius=MINE_SIDE, color=(255, 255, 255), thickness=3)   # 雷区边线
        cv2.circle(_map, center, radius=MINE_RADIUS, color=(0, 255, 255), thickness=-1)    # 雷区中心
    return _map

def draw_entity(map, tracker, r2):
    cv2.circle(map, (tracker.self_lidar_x, tracker.self_lidar_y),
                       5, (255, 0, 255), -1)   # 车辆中心
    # 车体对地角度
    cv2.line(
        map,
        (tracker.self_lidar_x, tracker.self_lidar_y),
        (tracker.self_lidar_x + int(100 * math.cos(math.radians(tracker.prev_angle) + math.pi / 2)),
            tracker.self_lidar_y + int(100 * math.sin(math.radians(tracker.prev_angle) - math.pi / 2))),
        (255, 0, 0), 2)

    # 云台对地角度 r2
    cv2.line(
        map,
        (tracker.self_lidar_x, tracker.self_lidar_y),
        (tracker.self_lidar_x + int(50 * math.cos(math.radians(r2+tracker.prev_angle) + math.pi / 2)),
            tracker.self_lidar_y + int(50 * math.sin(math.radians(r2+tracker.prev_angle) - math.pi / 2))),
        (0, 0, 255), 2)
    
    cv2.polylines(map, [tracker.car_box], True, (255, 0, 0), 2)
    cv2.drawMarker(map, (tracker.dest_x, tracker.dest_y), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
    cv2.rectangle(map, (tracker.dest_x, tracker.dest_y), (tracker.dest_x, tracker.dest_y), (0, 255, 0), 2)
    if tracker.targ_x is not None and tracker.targ_y is not None:
        cv2.drawMarker(map, (int(tracker.targ_x), int(tracker.targ_y)), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
        cv2.rectangle(map, (int(tracker.targ_x), int(tracker.targ_y)), (int(tracker.targ_x), int(tracker.targ_y)), (255, 0, 0), 2)
    
    cv2.putText(map, f"r2: {r2:.2f}", (tracker.self_lidar_x+10, tracker.self_lidar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 绘制规划路径
    if tracker.path:
        pts = np.array(tracker.path, dtype=np.int32)
        cv2.polylines(map, [pts], False, (0, 255, 255), 1)
    # 绘制当前最近路径点
    if tracker.path_point is not None:
        cv2.drawMarker(map, (int(tracker.path_point[0]), int(tracker.path_point[1])), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        # cv2.circle(map, (int(tracker.path_point[0]), int(tracker.path_point[1])), 4, (0, 255, 0), -1)
    return map

def map_dest_point(x):

    idx = bisect.bisect(THRESHOLDS, x) - 1
    if 0 <= idx < len(RESULT_DEST):
        return RESULT_DEST[idx]
    return None