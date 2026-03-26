#!/usr/bin/env python3
"""
cam_3 이미지 폴더 → ROS2 bag 변환

Usage:
  python3 scripts/create_bag.py \
      --image_dir test_data/records_data/cam_3/images \
      --output    output/cam3.bag \
      --topic     /cam_3/image \
      --encoding  bgr8
"""
import argparse, os, sys
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="test_data/records_data/cam_3/images")
    parser.add_argument("--output",    default="output/cam3.bag")
    parser.add_argument("--topic",     default="/cam_3/image")
    parser.add_argument("--encoding",  default="bgr8")
    parser.add_argument("--frame_id",  default="cam_3")
    parser.add_argument("--fps",       type=float, default=1.0,
                        help="Playback rate in Hz (default: 1.0 → 1 sec/frame)")
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png"}
    img_paths = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not img_paths:
        print(f"No images found in {args.image_dir}"); sys.exit(1)
    print(f"Found {len(img_paths)} images")

    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image
    from builtin_interfaces.msg import Time

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    writer = rosbag2_py.SequentialWriter()
    storage_opts = rosbag2_py.StorageOptions(uri=args.output, storage_id="sqlite3")
    conv_opts    = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer.open(storage_opts, conv_opts)
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=args.topic,
        type="sensor_msgs/msg/Image",
        serialization_format="cdr",
    ))

    interval_ns = int(1e9 / args.fps)   # nanoseconds per frame

    for i, path in enumerate(img_paths):
        # Always use index-based timestamps at the requested fps
        ts_ns      = i * interval_ns
        ts_sec     = ts_ns // 10**9
        ts_ns_part = ts_ns % 10**9

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"  Skip (load failed): {path}"); continue

        h, w = img_bgr.shape[:2]
        msg = Image()
        msg.header.stamp.sec     = ts_sec
        msg.header.stamp.nanosec = ts_ns_part
        msg.header.frame_id      = args.frame_id
        msg.height    = h
        msg.width     = w
        msg.encoding  = args.encoding
        msg.step      = w * 3
        msg.data      = img_bgr.tobytes()

        writer.write(args.topic, serialize_message(msg), ts_ns)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(img_paths)}")

    del writer
    print(f"\nBag saved: {args.output}")
    print(f"Topic: {args.topic}  |  {len(img_paths)} frames")


if __name__ == "__main__":
    main()
