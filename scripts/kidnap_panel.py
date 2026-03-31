#!/usr/bin/env python3
"""
kidnap_panel.py
===============
/kidnap_idx (Int32) 를 퍼블리시하는 소형 PyQt5 GUI 창.
kidnap_localizer_viewer.launch.py 에서 함께 실행됨.

단독 실행:
  python3 scripts/kidnap_panel.py
"""

import sys
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIntValidator


class IdxPublisher(Node):
    def __init__(self):
        super().__init__('kidnap_panel')
        self.pub = self.create_publisher(Int32, '/kidnap_idx', 10)

    def send(self, idx: int):
        msg = Int32()
        msg.data = idx
        self.pub.publish(msg)
        self.get_logger().info(f'Published /kidnap_idx = {idx}')


class KidnapPanel(QWidget):
    def __init__(self, ros_node: IdxPublisher):
        super().__init__()
        self.node = ros_node
        self.setWindowTitle('Kidnap Localizer')
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 제목 ──────────────────────────────────────────────────────
        title = QLabel('Kidnap Localizer Panel')
        title.setFont(QFont('Sans', 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # ── 인덱스 입력 + Go 버튼 ─────────────────────────────────────
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText('index (1 ~ N)')
        self.input.setValidator(QIntValidator(0, 99999))
        self.input.setFont(QFont('Mono', 14))
        self.input.returnPressed.connect(self._send)
        row.addWidget(self.input)

        btn_go = QPushButton('Go')
        btn_go.setFixedWidth(60)
        btn_go.setFont(QFont('Sans', 11, QFont.Bold))
        btn_go.clicked.connect(self._send)
        row.addWidget(btn_go)
        root.addLayout(row)

        # ── 목록 버튼 ─────────────────────────────────────────────────
        btn_list = QPushButton('List  (send 0)')
        btn_list.clicked.connect(lambda: self._publish(0))
        root.addWidget(btn_list)

        # ── ± 이동 버튼 ───────────────────────────────────────────────
        nav = QHBoxLayout()
        for label, delta in [('◀◀ -10', -10), ('◀ -1', -1),
                              ('+1 ▶',   1), ('+10 ▶▶', 10)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, d=delta: self._step(d))
            nav.addWidget(b)
        root.addLayout(nav)

        # ── 히스토리 ─────────────────────────────────────────────────
        root.addWidget(QLabel('History:'))
        self.history = QListWidget()
        self.history.setMaximumHeight(160)
        self.history.itemDoubleClicked.connect(self._replay)
        root.addWidget(self.history)

        # ── 상태 표시 ─────────────────────────────────────────────────
        self.status = QLabel('Ready')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet('color: gray; font-size: 10px;')
        root.addWidget(self.status)

    def _send(self):
        text = self.input.text().strip()
        if not text:
            return
        self._publish(int(text))

    def _step(self, delta):
        text = self.input.text().strip()
        cur  = int(text) if text else 1
        nxt  = max(1, cur + delta)
        self.input.setText(str(nxt))
        self._publish(nxt)

    def _publish(self, idx: int):
        self.node.send(idx)
        self.status.setText(f'→ {idx}')
        if idx > 0:
            self.input.setText(str(idx))
            item = QListWidgetItem(str(idx))
            self.history.insertItem(0, item)
            # 최대 30개 유지
            while self.history.count() > 30:
                self.history.takeItem(self.history.count() - 1)

    def _replay(self, item: QListWidgetItem):
        idx = int(item.text())
        self.input.setText(str(idx))
        self._publish(idx)


def main():
    rclpy.init()
    ros_node = IdxPublisher()

    # ROS2 spin → 별도 스레드
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    win = KidnapPanel(ros_node)
    win.show()

    # Qt가 살아 있는 동안 ROS keepalive
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    ret = app.exec_()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
