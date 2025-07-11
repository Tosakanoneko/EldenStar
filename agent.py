#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import zmq
import numpy as np
import datetime
from typing import cast
import json
import serial
import time
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math
import os
import subprocess
# ROS2 Launch 相关
from launch import LaunchService, LaunchDescription
from launch_ros.actions import Node as LaunchNode
from utils import PIDController

class EthClient:
    SRC_IP = "10.0.0.1"
    PORT   = 5555

    def __init__(self):
        ctx = zmq.Context(io_threads=2)      # 多 1 个 IO 线程
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM,   2)  # 收端也留 2 帧余量
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVTIMEO, 1)  # 1 ms 超时，防死等
        self.sock.connect(f"tcp://{self.SRC_IP}:{self.PORT}")

    def rcv_frame(self):
        try:
            # 由于 CONFLATE，只能收到最新一条
            buf = self.sock.recv(flags=0)  # ByteString | Frame
        except zmq.Again:
            return None

        # 为静态类型检查器显式断言类型，确保满足 buffer 协议 (bytes)
        buf_bytes = cast(bytes, buf)

        img = np.frombuffer(buf_bytes, dtype=np.uint8)
        return cv2.imdecode(img, cv2.IMREAD_COLOR)

    def close(self):
        self.sock.close()

class SlaveDeviceAgent:
    def __init__(self):
        self.send_data = {
            "dx": 0, # 相对值
            "dy": 0, # 相对值
            "px": 0, # 相对值
            "py": 0, # 相对值
            "dr1": 0, # 车体旋转角度，对地
        }
        self.recv_data = {
            "s1": 0,
            "s2": 0,
            "s3": 0,
            "s4": 0,
            "r2": 0,
        }
        self.ser = serial.Serial(port='/dev/ttyTHS1', baudrate=115200, timeout=0.01) # ttyCH341USB0
        self.lidar_searching_pid = PIDController(Kp=10, Ki=0, Kd=0, output_limits=(-1500, 1500))
        self.lidar_found = False

        self.short2enemy_pid = PIDController(Kp=1, Ki=0, Kd=0, output_limits=(100, 100))

    def send_json(self):
        while True:
            json_data = '@'
            json_data += json.dumps(self.send_data)
            json_data += '\r\n'
            self.ser.write(json_data.encode('utf-8'))
            # print(f"Sent: {json_data}")
            time.sleep(0.001)

    def recv_json(self):
        while True:
            json_data = self.ser.readline().decode('utf-8').strip()
            if json_data.startswith('@'):
                json_data = json_data[1:]
                try:
                    self.recv_data = json.loads(json_data)
                    # print(f"Received: {self.recv_data}")
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON: {json_data}")
            else:
                if json_data:  # 如果有非空数据但格式不对
                    print(f"Invalid data format: {json_data}")
            time.sleep(0.001)

    def lidar_searching(self, rel_angle):
        # print(rel_angle)
        # self.send_data["px"] = 0
        self.send_data["px"] = int(self.lidar_searching_pid.update(-rel_angle))
        self.send_data["py"] = 0
    
    def short2enemy(self, rel_car_angle):
        # self.send_data["dr1"] = rel_car_angle
        self.send_data["dr1"] = 180
    
    def close_sd(self):
        self.ser.close()

class LidarAgent(Node):
    def __init__(self):
        super().__init__('scan_visualization_agent')

        # ------------ 参数声明 ------------
        # 默认值 60.0，可在 launch 中或运行后用 ros2 param set 调整
        CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                _cfg = json.load(f)
                _default_remap = float(_cfg.get('remap_factor', 60.0))
        except Exception:
            _default_remap = 60.0
        self.declare_parameter('remap_factor', _default_remap)
        self.remap_factor = self.get_parameter(
            'remap_factor').get_parameter_value().double_value

        # ------------ 订阅 ------------
        self.create_subscription(
            LaserScan,
            '/scan',
            self.process_lidar_scan,
            10)
        self.create_subscription(
            Float32,
            '/remap_factor',
            self.remap_factor_callback,
            10)

        # ------------ 其他状态 ------------
        self.points = []
        self.calib = False

    # ---------- 参数 / 话题回调 ----------
    def remap_factor_callback(self, msg: Float32):
        self.remap_factor = msg.data
        # 同步更新节点参数，便于 ros2 param get 查询
        self.set_parameters([rclpy.Parameter(
            'remap_factor', rclpy.Parameter.Type.DOUBLE, self.remap_factor)])
        self.get_logger().info(f'边界系数更新为: {self.remap_factor}')

    def process_lidar_scan(self, scan: LaserScan):
        # scan数据->points(x,y)
        self.remap_factor = self.get_parameter(
            'remap_factor').value
        angle = scan.angle_min
        points = []
        for r in scan.ranges:
            if math.isinf(r):
                r = 0.0
            if r == 0.0 or self.remap_factor is None:
                x, y = 0, 0
            else:
                x = int(r * self.remap_factor * math.cos(angle + math.pi/2))
                y = int(r * self.remap_factor * math.sin(angle - math.pi/2))
                if abs(x) > 299 or abs(y) > 299:
                    x, y = 0, 0
            points.append((x, y))
            angle += scan.angle_increment
        self.points = points

def test_eth():
    client = EthClient()
    recording = False
    writer = None
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    while True:
        frame = client.rcv_frame()
        if recording and writer is not None:
            writer.write(frame)
        if frame is None:
            continue
        # frame = cv2.resize(frame, (640, 480))
        cv2.circle(frame, (320-39, 240-10), 3, (0, 255, 255), -1)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("frame.jpg", frame)
            print("Saved frame.jpg")
        elif key == ord('r'):
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
                writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
                recording = True
                print(f"Start recording {filename}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Stop recording")
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    client.close()

def test_cam():
    recording = False
    writer = None
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    while True:
        ret, frame = cap.read()
        if recording and writer is not None:
            writer.write(frame)
        if frame is None:
            continue
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("frame.jpg", frame)
            print("Saved frame.jpg")
        elif key == ord('r'):
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
                writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
                recording = True
                print(f"Start recording {filename}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Stop recording")
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    cap.release()

def test_ser():
    agent = SlaveDeviceAgent()
    send_json_thread = threading.Thread(target=agent.send_json)
    send_json_thread.daemon = True
    send_json_thread.start()

    recv_json_thread = threading.Thread(target=agent.recv_json)
    recv_json_thread.daemon = True
    recv_json_thread.start()

    while True:
        time.sleep(0.5)

def start_rplidar_node():
    subprocess.run(['pkill', '-f', 'rplidar_node'], stdout=subprocess.DEVNULL)

    """以 LaunchService 方式后台启动 rplidar_node."""
    ld = LaunchDescription()
    rplidar = LaunchNode(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': '/dev/ttyCH341USB0',  # 替换为实际端口（如有需要）
            'serial_baudrate': 115200,
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Sensitivity',
            'max_distance': 8,
        }],
        output='screen'
    )
    ld.add_action(rplidar)

    ls = LaunchService(argv=[])
    ls.include_launch_description(ld)
    ls.run()

if __name__ == "__main__":
    test_eth()