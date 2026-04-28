import pickle
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import Path
import numpy as np
import math

OUTLIER_THRESH = 5.0  # eval_trajectory.py 와 동일한 기준 (m)

class VPSPublisher(Node):
    def __init__(self):
        super().__init__('vps_publisher')

        self.pose_pub = self.create_publisher(PoseStamped, '/vps/current_pose', 4)
        self.error_pub = self.create_publisher(String, '/vps/error', 10)
        self.pred_path_pub = self.create_publisher(Path, '/vps/pred_path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/vps/gt_path', 10)

        self.timer = self.create_timer(0.25, self.timer_callback)

        # T_align 로드: GS 학습 시 사용한 바닥 정렬 변환 (Kapture world → GS world)
        step0_pkl = '/home/park/loc_ws/src/render_loc/output/sgs_multi_cam_test/step0_data.pkl'
        step0_data = pickle.load(open(step0_pkl, 'rb'))
        self.T_align = np.array(step0_data['T_align'], dtype=np.float64)

        pred_csv = '/home/park/loc_ws/src/render_loc/output/sgs_multi_cam_test/test_results/cam_3/trajectory.csv'
        gt_path  = '/home/park/loc_ws/src/render_loc/kapture/sensors/trajectories.txt'

        # GT: Kapture world → GS world 변환
        gt_poses_kapture = self.load_gt_trajectory(gt_path)
        self.gt_poses = {ts: self.apply_T_align(pose) for ts, pose in gt_poses_kapture.items()}

        # Pred: success=1 로드 후 GT 대비 outlier 분류
        self.pred_poses = self.load_csv_trajectory(pred_csv)
        self.pred_is_inlier = self._classify_inliers(self.pred_poses)

        n_inlier  = sum(self.pred_is_inlier)
        n_outlier = len(self.pred_is_inlier) - n_inlier
        self.get_logger().info(
            f"VPS Publisher 시작 | pred={len(self.pred_poses)} (inlier={n_inlier}, outlier={n_outlier})")

        # Path 시각화: inlier 프레임 위치만 사용
        inlier_poses = [p for p, ok in zip(self.pred_poses, self.pred_is_inlier) if ok]
        self.pred_path_msg = self.create_path_msg(inlier_poses)

        gt_list = [[t] + self.gt_poses[t] for t in sorted(self.gt_poses)]
        self.gt_path_msg = self.create_path_msg(gt_list)

        self.current_idx = 0
        self.last_inlier_pose = None  # outlier 시 fallback용

        # Roll 보정 없음 — GT orientation 자체를 그대로 사용
        # 뷰가 틀어져 있으면 아래 주석 해제해서 조정:
        #   왼쪽 90° : [0, 0, +0.7071, 0.7071]
        #   오른쪽 90°: [0, 0, -0.7071, 0.7071]
        #   180°      : [0, 0,  1.0,    0.0   ]
        self._q_roll = [0.0, 0.0, 0.0, 1.0]  # identity (no roll)

    def _classify_inliers(self, pred_poses):
        """GT 대비 translation error < OUTLIER_THRESH 이면 inlier."""
        flags = []
        for data in pred_poses:
            ts, pred_pose = data[0], data[1:]
            if ts in self.gt_poses:
                gt_xyz   = np.array(self.gt_poses[ts][:3])
                pred_xyz = np.array(pred_pose[:3])
                is_inlier = np.linalg.norm(pred_xyz - gt_xyz) < OUTLIER_THRESH
            else:
                is_inlier = True  # GT 없으면 outlier 판단 불가 → inlier로 간주
            flags.append(is_inlier)
        return flags

    def load_csv_trajectory(self, file_path):
        # CSV: timestamp,tx,ty,tz,qx,qy,qz,qw,success  — success=1만 로드
        poses = []
        with open(file_path, 'r') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9 or parts[-1].strip() != '1':
                    continue
                ts, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts[:8])
                poses.append([ts, tx, ty, tz, qx, qy, qz, qw])
        return poses

    def load_gt_trajectory(self, file_path):
        # Kapture format: timestamp(us), device_id, qw, qx, qy, qz, tx, ty, tz  (comma-separated)
        gt_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                if 'cam_3' not in line:
                    continue
                parts = [p.strip() for p in line.strip().split(',')]
                numeric_parts = []
                for p in parts:
                    try:
                        numeric_parts.append(float(p))
                    except ValueError:
                        pass
                if len(numeric_parts) >= 8:
                    timestamp_s = numeric_parts[0] / 1e6
                    qw, qx, qy, qz = numeric_parts[1], numeric_parts[2], numeric_parts[3], numeric_parts[4]
                    tx, ty, tz     = numeric_parts[5], numeric_parts[6], numeric_parts[7]
                    gt_dict[timestamp_s] = [tx, ty, tz, qx, qy, qz, qw]
        return gt_dict

    def apply_T_align(self, pose):
        tx, ty, tz, qx, qy, qz, qw = pose
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = R
        T_c2w[:3, 3]  = [tx, ty, tz]
        T_c2w_gs = self.T_align @ T_c2w
        new_tx, new_ty, new_tz = T_c2w_gs[:3, 3]
        new_qx, new_qy, new_qz, new_qw = self._rotmat_to_quat(T_c2w_gs[:3, :3])
        return [new_tx, new_ty, new_tz, new_qx, new_qy, new_qz, new_qw]

    @staticmethod
    def _rotmat_to_quat(R):
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2,1] - R[1,2]) * s
            qy = (R[0,2] - R[2,0]) * s
            qz = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
        return qx, qy, qz, qw

    def create_path_msg(self, pose_list):
        path = Path()
        path.header.frame_id = "map"
        for data in pose_list:
            p = PoseStamped()
            p.pose.position.x = data[1]
            p.pose.position.y = data[2]
            p.pose.position.z = data[3]
            path.poses.append(p)
        return path

    @staticmethod
    def _quat_mul(q1, q2):
        """Hamilton product q1 * q2. 포맷: [x, y, z, w]."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ]

    def calculate_error(self, pred, gt):
        trans_err = np.linalg.norm(np.array(pred[:3]) - np.array(gt[:3]))
        p_quat = np.array(pred[3:])
        g_quat = np.array(gt[3:])
        dot    = np.clip(np.abs(np.dot(p_quat, g_quat)), -1.0, 1.0)
        rot_err_deg = math.degrees(2 * math.acos(dot))
        return trans_err, rot_err_deg

    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        self.pred_path_msg.header.stamp = now
        self.gt_path_msg.header.stamp = now
        self.pred_path_pub.publish(self.pred_path_msg)
        self.gt_path_pub.publish(self.gt_path_msg)

        if self.current_idx >= len(self.pred_poses):
            return

        pred_data  = self.pred_poses[self.current_idx]
        timestamp  = pred_data[0]
        pred_pose  = pred_data[1:]
        is_inlier  = self.pred_is_inlier[self.current_idx]
        self.current_idx += 1

        # inlier면 포즈 갱신, outlier면 마지막 inlier 포즈 유지
        if is_inlier:
            self.last_inlier_pose = pred_pose
        pose_to_pub = self.last_inlier_pose if self.last_inlier_pose is not None else pred_pose

        msg = PoseStamped()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.pose.position.x    = pose_to_pub[0]
        msg.pose.position.y    = pose_to_pub[1]
        msg.pose.position.z    = pose_to_pub[2]
        # orientation: GT 사용 (안정적), position: pred 사용 (VPS 추정값)
        if timestamp in self.gt_poses:
            q = self.gt_poses[timestamp][3:7]
        else:
            q = pose_to_pub[3:7]
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pose_pub.publish(msg)

        err_msg = String()
        if timestamp in self.gt_poses:
            gt_pose = self.gt_poses[timestamp]
            t_err, r_err = self.calculate_error(pred_pose, gt_pose)
            status = "inlier" if is_inlier else "OUTLIER(hold)"
            err_msg.data = (f"Frame: {self.current_idx} [{status}] | "
                            f"Trans Err: {t_err:.4f} m | Rot Err: {r_err:.2f} deg")
        else:
            err_msg.data = f"Frame: {self.current_idx} | Trans Err: N/A | Rot Err: N/A (GT 미매칭)"
        self.error_pub.publish(err_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VPSPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
