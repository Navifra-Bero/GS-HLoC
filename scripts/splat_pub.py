import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import os

class SplatPublisher(Node):
    def __init__(self):
        super().__init__('splat_publisher')
        
        # 파일 경로 설정
        self.declare_parameter('splat_file', '/home/park/loc_ws/src/render_loc/output/gs_test/gaussian/gaussian_map.splat')
        splat_path = self.get_parameter('splat_file').get_parameter_value().string_value
        
        self.publisher = self.create_publisher(PointCloud2, '/vps/gaussian_map', 10)
        
        if not os.path.exists(splat_path):
            self.get_logger().error(f"파일을 찾을 수 없습니다: {splat_path}")
            return

        self.get_logger().info(f"Splat 로드 및 변환 중: {splat_path}")
        self.pc2_msg = self.prepare_pc2_message(splat_path)
        
        # 1초마다 반복 발행 (RViz에서 지속적으로 확인 가능)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("발행 준비 완료")

    def load_splat(self, path):
        # .splat 파일 포맷: 가우시안당 32바이트
        data = np.fromfile(path, dtype=np.uint8)
        num_splats = len(data) // 32
        
        # 성능을 위해 데이터 솎아내기 (너무 느리면 사용, 예: 2개당 1개)
        # data = data.reshape(num_splats, 32)[::2].flatten()
        # num_splats = len(data) // 32

        reshaped = data.reshape(num_splats, 32)
        
        # XYZ 추출 (float32, 12바이트)
        xyz = reshaped[:, 0:12].view(np.float32)
        
        # RGBA 추출 (uint8, 24~28바이트)
        # RViz PointCloud2의 RGB 패킹 포맷에 맞춰 정렬 (BGRA 혹은 RGBA)
        rgb = reshaped[:, 24:27] # R, G, B만 사용
        
        return xyz, rgb

    def prepare_pc2_message(self, path):
        xyz, rgb = self.load_splat(path)
        num_points = xyz.shape[0]

        # PointCloud2를 위한 구조화된 배열 생성
        # x, y, z (float32) + rgb (uint32로 패킹)
        buffer = np.zeros(num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        buffer['x'] = xyz[:, 0]
        buffer['y'] = xyz[:, 1]
        buffer['z'] = xyz[:, 2]

        # RGB 값을 하나의 uint32로 패킹 (R << 16 | G << 8 | B)
        # 이 작업은 NumPy 연산으로 한 번에 처리하여 속도를 높입니다.
        rgb_packed = (rgb[:, 0].astype(np.uint32) << 16) | \
                     (rgb[:, 1].astype(np.uint32) << 8) | \
                      rgb[:, 2].astype(np.uint32)
        buffer['rgb'] = rgb_packed

        # Header 설정 (오류 수정 지점)
        header = Header()
        header.frame_id = 'map'
        
        # 필드 정의
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # PointCloud2 메시지 생성
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = num_points
        msg.is_dense = False
        msg.is_bigendian = False
        msg.fields = fields
        msg.point_step = 16 # (float32*3 + uint32*1)
        msg.row_step = msg.point_step * num_points
        msg.data = buffer.tobytes()

        return msg

    def timer_callback(self):
        self.pc2_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SplatPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()