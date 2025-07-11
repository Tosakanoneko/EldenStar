import cv2
import multiprocessing
from agent import SlaveDeviceAgent, EthClient, start_rplidar_node
from det import Detector
import threading
import rclpy
from agent import LidarAgent
from utils import *
from nav import PPTracker
import math

def main():
    rclpy.init()
    lidar_agent   = LidarAgent()
    pptracker = PPTracker()
    detector = Detector()
    client = EthClient()
    sd_agent = SlaveDeviceAgent()
    # 启动 rplidar_node 独立进程
    lidar_proc = multiprocessing.Process(target=start_rplidar_node, daemon=True)
    lidar_proc.start()

    cam_debug = True
    lidar_debug = False

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    if cam_debug:
        cv2.namedWindow("pid", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Kp_x*10", "pid", int(detector.pid_far_x.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_x*10", "pid", int(detector.pid_far_x.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_x*10", "pid", int(detector.pid_far_x.Kd*10), 200, lambda x: None)
        cv2.createTrackbar("Kp_y*10", "pid", int(detector.pid_far_y.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_y*10", "pid", int(detector.pid_far_y.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_y*10", "pid", int(detector.pid_far_y.Kd*10), 200, lambda x: None)
    if lidar_debug:
        cv2.namedWindow("pid", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Kp_x*10", "pid", int(sd_agent.lidar_searching_pid.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_x*10", "pid", int(sd_agent.lidar_searching_pid.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_x*10", "pid", int(sd_agent.lidar_searching_pid.Kd*10), 200, lambda x: None)
        cv2.createTrackbar("Kp_y*10", "pid", int(sd_agent.lidar_searching_pid.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_y*10", "pid", int(sd_agent.lidar_searching_pid.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_y*10", "pid", int(sd_agent.lidar_searching_pid.Kd*10), 200, lambda x: None)

    stage = 0 


    send_json_thread = threading.Thread(target=sd_agent.send_json)
    send_json_thread.daemon = True
    send_json_thread.start()

    recv_json_thread = threading.Thread(target=sd_agent.recv_json)
    recv_json_thread.daemon = True
    recv_json_thread.start()

    while True:
        frame = client.rcv_frame()
        if frame is None:
            continue

        if pptracker.targ_x is not None and pptracker.targ_y is not None:
            dist = math.sqrt((pptracker.targ_x-pptracker.self_lidar_x)**2 + (pptracker.targ_y-pptracker.self_lidar_y)**2)
        else:
            dist = 300
        frame = detector.track_point(frame, dist)


        if cam_debug:
            detector.pid_far_x.Kp = cv2.getTrackbarPos("Kp_x*10", "pid") / 10
            detector.pid_far_x.Ki = cv2.getTrackbarPos("Ki_x*10", "pid") / 10
            detector.pid_far_x.Kd = cv2.getTrackbarPos("Kd_x*10", "pid") / 10
            detector.pid_far_y.Kp = cv2.getTrackbarPos("Kp_y*10", "pid") / 10
            detector.pid_far_y.Ki = cv2.getTrackbarPos("Ki_y*10", "pid") / 10
            detector.pid_far_y.Kd = cv2.getTrackbarPos("Kd_y*10", "pid") / 10
        if lidar_debug:
            sd_agent.lidar_searching_pid.Kp = cv2.getTrackbarPos("Kp_x*10", "pid") / 10
            sd_agent.lidar_searching_pid.Ki = cv2.getTrackbarPos("Ki_x*10", "pid") / 10
            sd_agent.lidar_searching_pid.Kd = cv2.getTrackbarPos("Kd_x*10", "pid") / 10
            sd_agent.lidar_searching_pid.Kp = cv2.getTrackbarPos("Kp_y*10", "pid") / 10
            sd_agent.lidar_searching_pid.Ki = cv2.getTrackbarPos("Ki_y*10", "pid") / 10
            sd_agent.lidar_searching_pid.Kd = cv2.getTrackbarPos("Kd_y*10", "pid") / 10

        rclpy.spin_once(lidar_agent, timeout_sec=0.0)

        point_cloud = pptracker.map_cloud(lidar_agent.points)
        lidar_processed = pptracker.navi(point_cloud)
        print("r2", sd_agent.recv_data["r2"])
        abs_gimble = float(sd_agent.recv_data["r2"])+pptracker.prev_angle
        while abs_gimble > 180:
            abs_gimble -= 360
        while abs_gimble <= -180:
            abs_gimble += 360
        pptracker.abs_gimble = abs_gimble
        pptracker.abs_self2enemyR = pptracker.get_relative_angle(pptracker.prev_angle)+pptracker.prev_angle
        pptracker.rel_gimble2enemyR = pptracker.abs_self2enemyR-pptracker.abs_gimble

        # sd_agent.short2enemy(pptracker.abs_self2enemyR)
        if pptracker.self_lidar_y < 299:
            sd_agent.send_data["dr1"] =175
        else:
            sd_agent.send_data["dr1"] = 0

        if pptracker.path_point is not None:
            sd_agent.send_data["dx"] = (pptracker.path_point[0] - pptracker.self_lidar_x)*-1
            sd_agent.send_data["dy"] = (pptracker.path_point[1] - pptracker.self_lidar_y)*-1
            print("dx,dy,dr1", sd_agent.send_data["dx"], sd_agent.send_data["dy"], sd_agent.send_data["dr1"])
        else:
            sd_agent.send_data["dx"] = 0
            sd_agent.send_data["dy"] = 0
        
        if detector.missing:
            sd_agent.lidar_searching(pptracker.rel_gimble2enemyR)
        else:
            sd_agent.lidar_searching_pid.reset()
            sd_agent.send_data["px"] = detector.dpx  # 误差
            sd_agent.send_data["py"] = detector.dpy  # 误差
        
        
        lidar_processed = cv2.resize(lidar_processed, (300, 300))
        vis_map = pptracker._map.copy()
        vis_map = draw_entity(vis_map, pptracker, pptracker.abs_gimble)
        vis_map = cv2.resize(vis_map, (360, 360))

        frame = cv2.resize(frame, (480, 360))
        combined = np.hstack((frame, vis_map))
        cv2.imshow("frame", combined)
        if cv2.waitKey(1) == 27:
            break

    # 释放资源
    client.close()
    cv2.destroyAllWindows()

    # 结束 rplidar_node 子进程
    if lidar_proc.is_alive():
        lidar_proc.terminate()
        lidar_proc.join()
    rclpy.shutdown()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()