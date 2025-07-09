import cv2
import multiprocessing
from agent import SlaveDeviceAgent, EthClient, start_rplidar_node
from det import Detector
import threading
import rclpy
from agent import LidarAgent
from utils import *
from nav import PPTracker

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

    cam_debug = False
    lidar_debug = True

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("lidar", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('lidar', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.namedWindow("pid", cv2.WINDOW_NORMAL)

    if cam_debug:
        cv2.createTrackbar("Kp_x*10", "pid", int(detector.pid_x.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_x*10", "pid", int(detector.pid_x.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_x*10", "pid", int(detector.pid_x.Kd*10), 200, lambda x: None)
        cv2.createTrackbar("Kp_y*10", "pid", int(detector.pid_y.Kp*10), 200, lambda x: None)
        cv2.createTrackbar("Ki_y*10", "pid", int(detector.pid_y.Ki*10), 200, lambda x: None)
        cv2.createTrackbar("Kd_y*10", "pid", int(detector.pid_y.Kd*10), 200, lambda x: None)
    if lidar_debug:
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
        # frame = cv2.imread("white.jpg")
        # frame = cv2.resize(frame, (640, 480))
        if frame is None:
            pass

        frame = detector.process_frame(frame)


        if cam_debug:
            detector.pid_x.Kp = cv2.getTrackbarPos("Kp_x*10", "pid") / 10
            detector.pid_x.Ki = cv2.getTrackbarPos("Ki_x*10", "pid") / 10
            detector.pid_x.Kd = cv2.getTrackbarPos("Kd_x*10", "pid") / 10
            detector.pid_y.Kp = cv2.getTrackbarPos("Kp_y*10", "pid") / 10
            detector.pid_y.Ki = cv2.getTrackbarPos("Ki_y*10", "pid") / 10
            detector.pid_y.Kd = cv2.getTrackbarPos("Kd_y*10", "pid") / 10
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


        lidar_processed = cv2.resize(lidar_processed, (300, 300))

        # combined = np.hstack((lidar_processed, vis_map))
        # combined = cv2.resize(combined, (600, 300))

        # sd_agent.send_data["dx"] = pptracker.dest_x - pptracker.self_lidar_x
        # sd_agent.send_data["dy"] = pptracker.dest_y - pptracker.self_lidar_y
        # print("dx,dy", sd_agent.send_data["dx"], sd_agent.send_data["dy"])
        if detector.missing:
            print("lidar searching")
            sd_agent.lidar_searching(pptracker.get_relative_angle())
        else:
            sd_agent.send_data["px"] = detector.dpx  # 误差
            sd_agent.send_data["py"] = detector.dpy  # 误差
        # sd_agent.send_data["px"] = 0
        # sd_agent.send_data["py"] = 0
        print(sd_agent.send_data)
        
        vis_map = pptracker._map.copy()
        vis_map = draw_entity(vis_map, pptracker)
        vis_map = cv2.resize(vis_map, (360, 360))

        frame = cv2.resize(frame, (480, 360))
        combined = np.hstack((frame, vis_map))
        cv2.imshow("frame", combined)
        # cv2.imshow("lidar", combined)
        if cv2.waitKey(1) == 27:
            break
        # elif cv2.waitKey(1) & 0xFF == ord('s'):

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