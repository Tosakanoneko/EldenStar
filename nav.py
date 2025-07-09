import cv2
import numpy as np
import math
import threading
import rclpy
import multiprocessing
from agent import start_rplidar_node, SlaveDeviceAgent
from agent import LidarAgent
from utils import *
import bisect

class PPTracker:
    def __init__(self):
        self.prev_angle = 0.0
        self.bias_angle = None         # 初始基准角
        self.ref_side   = None         # 参考边长
        self.prev_corners = None
        self.prev_center = None
        self.center_alpha = 0.3   # 中心位置滤波系数
        self.self_lidar_x, self.self_lidar_y = 300, 300
        self.targ_x, self.targ_y = 300, 300
        self.dest_x, self.dest_y = 0, 0
        self.wall_width = -20

        # 根据雷达安装位置调整
        self.front_offset_deg = 0   # 正值 = 逆时针
        self.car_length = 44
        self.car_width = 42
        self.car_box = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)

        # ---------------- Kalman ----------------
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix     = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix    = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        # 提高 processNoiseCov（1e-4 → 1e-3），降低 measurementNoiseCov（1.0 → 1e-2），让模型更信任新观测、收敛更快。
        self.kf.processNoiseCov      = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov  = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost         = np.eye(4, dtype=np.float32) * 0.1
        self.kf.statePost            = np.array([[self.self_lidar_x],
                                                 [self.self_lidar_y],
                                                 [0.0],
                                                 [0.0]], dtype=np.float32)

        self._map = draw_map()
        self._point_cloud_bg = np.zeros((600, 600), np.uint8)

    def clear(self):
        self.prev_angle = 0.0
        self.bias_angle = None
        self.ref_side   = None
        self.prev_corners = None
        self.prev_center = None
        self.kf.statePost = np.array([[300.0], [300.0], [0.0], [0.0]], dtype=np.float32)

    # -------------------- 几何检测 --------------------
    def _fit_rotated_square(self, gray):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k, iterations=1)

        

        _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(thresh == 255)
        if len(xs) == 0:
            return None, None, None
        pts = np.column_stack((xs, ys)).astype(np.float32)
        hull_indices = cv2.convexHull(pts, returnPoints=False)
        hull_pts = pts[hull_indices[:, 0]]

        # 点数过少，退化为轴对齐正方形
        if len(hull_pts) < 2:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            side = max(x_max - x_min, y_max - y_min)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            corners = np.array([[cx - side/2, cy - side/2],
                                [cx - side/2, cy + side/2],
                                [cx + side/2, cy + side/2],
                                [cx + side/2, cy - side/2]], dtype=np.float32)
            return corners, 0.0, side

        # 穷举边法线，寻找最小包围正方形
        min_area = float("inf")
        best_corners, best_angle_deg, best_side = None, 0.0, None
        for i in range(len(hull_pts)):
            p1, p2 = hull_pts[i], hull_pts[(i + 1) % len(hull_pts)]
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, s], [-s, c]], dtype=np.float32)
            rot = hull_pts @ R.T
            x0, x1 = rot[:, 0].min(), rot[:, 0].max()
            y0, y1 = rot[:, 1].min(), rot[:, 1].max()
            w, h = x1 - x0, y1 - y0
            side = max(w, h)
            if w >= h:                      # 横向为长边
                nx0, nx1 = x0, x0 + side
                yc = (y0 + y1) / 2.0
                ny0, ny1 = yc - side / 2, yc + side / 2
            else:                           # 纵向为长边
                ny0, ny1 = y0, y0 + side
                xc = (x0 + x1) / 2.0
                nx0, nx1 = xc - side / 2, xc + side / 2
            square_rot = np.array([[nx0, ny0], [nx0, ny1],
                                   [nx1, ny1], [nx1, ny0]], dtype=np.float32)
            R_inv = np.array([[c, -s], [s,  c]], dtype=np.float32)
            corners = square_rot @ R_inv.T
            if side * side < min_area:
                min_area = side * side
                best_corners = corners.copy()
                best_side = side
                deg = angle * 180.0 / math.pi
                while deg >= 90:  deg -= 180
                while deg <  -90: deg += 180
                best_angle_deg = deg
        return best_corners, best_angle_deg, best_side

    # 局部坐标系（正向 = 原正方形的左＋上）
    def _local_coord(self, corners_un, pt_un):
        idx = np.argmax(corners_un[:, 0] + corners_un[:, 1])   # 右下角作原点
        origin = corners_un[idx]
        n1, n2 = corners_un[(idx + 1) % 4] - origin, corners_un[(idx - 1) % 4] - origin
        vx, vy = (n1, n2) if abs(n1[0]) > abs(n1[1]) else (n2, n1)
        if vx[0] > 0: vx = -vx     # vx 指向"左"
        if vy[1] > 0: vy = -vy     # vy 指向"上"
        side = np.linalg.norm(vx)
        ux, uy = vx/side, vy/side
        rel = pt_un - origin
        local_x, local_y = np.dot(rel, ux), np.dot(rel, uy)
        mapped_x = max(0, min(600, local_x / side * 600))
        mapped_y = max(0, min(600, local_y / side * 600))
        return int(mapped_x), int(mapped_y)

    # -------------------- 主接口 --------------------
    def navi(self, gray):
        """
        输入：灰度图
        输出：可视化图像
        """
        corners_raw, raw_angle, raw_side = self._fit_rotated_square(gray)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners_raw is None or raw_side is None or raw_angle is None:
            return vis

        # ------ 大小滤波 ------
        is_valid_size = False
        if raw_side is not None:
            if self.ref_side is None:
                self.ref_side = raw_side
                is_valid_size = True
            elif 0.7 * self.ref_side <= raw_side <= 1.3 * self.ref_side:
                self.ref_side = 0.9 * self.ref_side + 0.1 * raw_side
                is_valid_size = True

        if is_valid_size:
            corners = corners_raw
            self.prev_corners = corners_raw
        else:
            corners = self.prev_corners if self.prev_corners is not None else corners_raw

        # ------ 位置滤波（指数平滑） ------
        if corners is not None:
            curr_center = corners.mean(axis=0)
            if self.prev_center is None:
                self.prev_center = curr_center
            smoothed_center = self.center_alpha * curr_center + (1 - self.center_alpha) * self.prev_center

            # 将平滑后的中心应用到角点（仅平移，不改变形状）
            shift = smoothed_center - curr_center
            corners = corners + shift

            # 更新缓存
            self.prev_center = smoothed_center
            self.prev_corners = corners

        # ------ 角度连续化 ------
        angle = raw_angle
        while angle >=  90: angle -= 180
        while angle <  -90: angle += 180
        while abs(angle - self.prev_angle) > 45:
            angle += -90 if angle > self.prev_angle else 90

        if self.bias_angle is None:       # 仅在第一次锁基准
            self.bias_angle = angle

        rel_rad = math.radians(angle - self.bias_angle + self.front_offset_deg)
        cr, sr = math.cos(-rel_rad), math.sin(-rel_rad)
        R_un = np.array([[cr, -sr], [sr, cr]], dtype=np.float32)
        corners_un = corners @ R_un.T
        pt_un      = np.dot(np.array([300., 300.], dtype=np.float32), R_un.T)

        mx_raw, my_raw = self._local_coord(corners_un, pt_un)
        mx_meas, my_meas = 599 - mx_raw, 599 - my_raw  # 取反映射

        # ------ Kalman 预测 + 更新 ------
        predicted = self.kf.predict()
        measurement = np.array([[np.float32(mx_meas)], [np.float32(my_meas)]])
        self.kf.correct(measurement)
        self.self_lidar_x, self.self_lidar_y = int(predicted[0, 0]), int(predicted[1, 0])
        # 获取车辆框四顶点坐标
        rect = ((self.self_lidar_x, self.self_lidar_y), (self.car_width, self.car_length), -self.prev_angle)
        self.car_box = cv2.boxPoints(rect)      # 计算 4 个顶点坐标（float32）
        self.car_box = np.int32(self.car_box)             # 转为整数坐标

        # ------ 可视化 ------
        # 绘制检测区域
        if corners is not None and self.wall_width != 0:
            center = corners.mean(axis=0)
            side_len = np.linalg.norm(corners[1] - corners[0]) if corners.shape[0] >= 2 else 0
            if side_len > 0:
                scale = (side_len + 2 * self.wall_width) / side_len
                detect_corners = (corners - center) * scale + center

                # ------- 雷达粗定位部分 -------
                mask_poly = detect_corners.astype(np.int32)
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.fillConvexPoly(mask, mask_poly, 255)

                region = cv2.bitwise_and(gray, mask)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                dilated = cv2.dilate(region, kernel, iterations=1)
                dilated = cv2.bitwise_and(dilated, mask)

                num_labels, labels = cv2.connectedComponents(dilated)

                if num_labels > 1:
                    sizes = [np.sum(labels == i) for i in range(1, num_labels)]
                    max_idx = int(np.argmax(sizes)) + 1

                    ys, xs = np.where(labels == max_idx)
                    if xs.size > 0:
                        targ_x, targ_y = int(xs.mean()), int(ys.mean())
                        pt_un_targ = np.dot(np.array([targ_x, targ_y], dtype=np.float32), R_un.T)
                        raw_tx, raw_ty = self._local_coord(corners_un, pt_un_targ)
                        self.targ_x, self.targ_y = 599 - raw_tx, 599 - raw_ty  # 与 self.self_lidar_x 同坐标系
                        
                        self.dest_x = map_dest_point(self.targ_x)
                        # self.dest_x = 149
                        self.dest_y = 225 if self.targ_y > 299 else 375
                        # self.dest_y = 375

                        cv2.drawMarker(vis, (int(xs.mean()), int(ys.mean())), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
                        cv2.rectangle(vis, (xs.min(), ys.min()), (xs.max(), ys.max()), (255, 0, 0), 2)
                       

                # 绘制检测区域边框
                cv2.polylines(vis, [mask_poly], True, (0, 255, 0), 2)

        cv2.polylines(vis, [np.array(corners, dtype=np.int32)], True, (0, 0, 255), 2)
        # cv2.putText(vis, f"raw: ({mx_meas:.1f},{my_meas:.1f})", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, f"self_x,self_y : ({self.self_lidar_x:.1f},{self.self_lidar_y:.1f})", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        # cv2.putText(vis, f"side:{self.ref_side:.1f}  bias:{angle:.1f}", (10, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"dest:({self.dest_x:.1f},{self.dest_y:.1f})", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(vis, (299, 299), 5, (0, 255, 255), -1)

        self.prev_angle = angle
        return vis

    def map_cloud(self, points):
        point_cloud = self._point_cloud_bg  # 直接复用底图，避免拷贝
        point_cloud.fill(0)                    # 清空上一帧

        if points:
            pts = np.asarray(points, dtype=np.int16)
            mask = (pts[:, 0] != 0) | (pts[:, 1] != 0)
            pts = pts[mask]
            if pts.size:
                xs = 300 + pts[:, 0]
                ys = 300 + pts[:, 1]
                np.clip(xs, 0, 599, out=xs)
                np.clip(ys, 0, 599, out=ys)
                point_cloud[ys, xs] = 255
        point_cloud[289:310, 289:310] = 0

        return point_cloud
    def get_relative_angle(self):
        """返回目标点相对车头的旋转角度（顺时针 0~180°, 逆时针 0~-180°）"""
        dx = self.targ_x - self.self_lidar_x
        dy = self.targ_y - self.self_lidar_y
        if dx == 0 and dy == 0:
            return 0

        bearing = math.degrees(math.atan2(-dy, dx))

        heading = self.prev_angle

        rel_angle = bearing - heading - 90

        while rel_angle > 180:
            rel_angle -= 360
        while rel_angle <= -180:
            rel_angle += 360

        return rel_angle

def main():
    rclpy.init()
    lidar_proc = multiprocessing.Process(target=start_rplidar_node, daemon=True)
    lidar_proc.start()

    sd_agent = SlaveDeviceAgent()

    send_json_thread = threading.Thread(target=sd_agent.send_json)
    send_json_thread.daemon = True
    send_json_thread.start()

    recv_json_thread = threading.Thread(target=sd_agent.recv_json)
    recv_json_thread.daemon = True
    recv_json_thread.start()


    lidar_agent   = LidarAgent()
    pptracker = PPTracker()

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 处理一次雷达回调（非阻塞）
            rclpy.spin_once(lidar_agent, timeout_sec=0.0)

            point_cloud = pptracker.map_cloud(lidar_agent.points)

            frame = pptracker.navi(point_cloud)

            print(pptracker.get_relative_angle())
            # agent.send_packet(data)

            vis_map = pptracker._map.copy()
            vis_map = draw_entity(vis_map, pptracker)
            # sd_agent.send_data["dx"] = pptracker.dest_x - pptracker.self_lidar_x
            # sd_agent.send_data["dy"] = pptracker.dest_y - pptracker.self_lidar_y
            sd_agent.send_data["dx"] = 0
            sd_agent.send_data["dy"] = 0
            # print("dx,dy", sd_agent.send_data["dx"], sd_agent.send_data["dy"])
            sd_agent.send_data["px"] = 0
            sd_agent.send_data["py"] = 0

            combined = np.hstack((frame, vis_map))
            combined = cv2.resize(combined, (750, 375))
            cv2.imshow('frame', combined)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            key = cv2.waitKey(1)
            if key == 27:                       # Esc → 退出
                break
            elif key == ord('r') or lidar_agent.calib:
                pptracker.clear()
                lidar_agent.calib = False
                print('重置偏移角度')
            elif key == ord('s'):
                print('开始冲线')

    except KeyboardInterrupt:
        print('程序被中断，退出…')

    finally:
        cv2.destroyAllWindows()
        # 结束 rplidar_node 子进程
    if lidar_proc.is_alive():
        lidar_proc.terminate()
        lidar_proc.join()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

