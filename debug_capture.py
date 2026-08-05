# -*- coding: utf-8 -*-
"""
麻将牌面截图调试工具

热键：
    F2  立即截取全屏，保存到 input/ 目录
    F3  拖拽鼠标框选区域，松开后保存框选内容
    ESC 取消框选

用法：
    python debug_capture.py [monitor_index]
        monitor_index: 默认 1（主显示器）；多显示器环境可传 2、3…

保存规则：
    自动沿用 input/ 下已有编号顺延（如已有 1.png，则保存为 2.png），
    与示例图 input/1.png 保持同一套命名，方便导入自测。
"""

import os
import sys
import time

import cv2
import keyboard
import mss
import numpy as np
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
COOLDOWN = 0.6  # 秒，防止按住热键时连发重复截图

_last_shot_time = 0.0


class HotkeyBridge(QObject):
    """
    热键信号桥：keyboard 的回调运行在独立线程，不能直接操作 Qt 界面，
    因此由该对象的信号把事件排队投递到 Qt 主线程（跨线程信号为队列连接）。
    """
    f2 = pyqtSignal()
    f3 = pyqtSignal()
    esc = pyqtSignal()


class SelectionOverlay(QWidget):
    """全屏半透明拖拽框选，交互与 main.py 保持一致。"""
    selection_done = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.showFullScreen()

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing = False

        self.mask_color = QColor(0, 0, 0, 160)
        self.rect_color = QColor(0, 255, 0, 200)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.mask_color)

        if self.start_x is not None and self.end_x is not None:
            x = min(self.start_x, self.end_x)
            y = min(self.start_y, self.end_y)
            w = abs(self.start_x - self.end_x)
            h = abs(self.start_y - self.end_y)

            if w > 5 and h > 5:
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.fillRect(x, y, w, h, Qt.transparent)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

                pen = QPen(self.rect_color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(x, y, w, h)

                info = f"{w} x {h}"
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(x + 5, y - 10, info)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_x = event.x()
            self.start_y = event.y()
            self.end_x = self.start_x
            self.end_y = self.start_y
            self.drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_x = event.x()
            self.end_y = event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_x = event.x()
            self.end_y = event.y()
            self.drawing = False

            x = min(self.start_x, self.end_x)
            y = min(self.start_y, self.end_y)
            w = abs(self.start_x - self.end_x)
            h = abs(self.start_y - self.end_y)

            if w > 20 and h > 20:
                self.selection_done.emit(x, y, w, h)
            else:
                self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.selection_done.emit(0, 0, 0, 0)


def next_save_path():
    """按 input/ 下已有编号顺延，返回下一个保存路径。"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    numbers = []
    for name in os.listdir(SAVE_DIR):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".png" and stem.isdigit():
            numbers.append(int(stem))
    return os.path.join(SAVE_DIR, f"{max(numbers) + 1 if numbers else 1}.png")


def grab_bgr(monitor):
    """用 mss 截取指定显示器/区域，返回 BGR 图像。"""
    with mss.MSS() as sct:
        shot = sct.grab(monitor)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def save_shot(img_bgr, source):
    global _last_shot_time
    now = time.time()
    if now - _last_shot_time < COOLDOWN:
        return
    _last_shot_time = now

    path = next_save_path()
    cv2.imwrite(path, img_bgr)
    h, w = img_bgr.shape[:2]
    print(f"[保存] {source} -> {path} ({w}x{h})")


def on_f2(monitor_index):
    try:
        with mss.MSS() as sct:
            monitors = sct.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                print(f"警告：显示器索引 {monitor_index} 不存在，使用主显示器")
                monitor = monitors[1]
            else:
                monitor = monitors[monitor_index]
            shot = sct.grab(monitor)
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        save_shot(img, "F2 全屏截图")
    except Exception as e:
        print(f"F2 截图失败: {e}")


def on_f3(overlay_ref):
    if overlay_ref["overlay"] is not None:
        return
    overlay = SelectionOverlay()
    overlay_ref["overlay"] = overlay
    overlay.selection_done.connect(lambda x, y, w, h: on_region_done(overlay_ref, x, y, w, h))
    overlay.show()


def on_region_done(overlay_ref, x, y, w, h):
    overlay = overlay_ref["overlay"]
    if overlay is not None:
        # 断开信号，避免 close/deleteLater 触发 closeEvent 再次进入本回调
        try:
            overlay.selection_done.disconnect()
        except TypeError:
            pass
        overlay_ref["overlay"] = None
        overlay.close()
        overlay.deleteLater()
    if x == 0 and y == 0 and w == 0 and h == 0:
        print("已取消框选")
        return
    try:
        img = grab_bgr({"left": x, "top": y, "width": w, "height": h})
        save_shot(img, "F3 框选截图")
    except Exception as e:
        print(f"F3 截图失败: {e}")


def main():
    monitor_index = 1
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        monitor_index = int(sys.argv[1])

    app = QApplication(sys.argv)
    bridge = HotkeyBridge()
    overlay_ref = {"overlay": None}

    bridge.f2.connect(lambda: on_f2(monitor_index))
    bridge.f3.connect(lambda: on_f3(overlay_ref))
    bridge.esc.connect(lambda: cancel_overlay(overlay_ref))

    try:
        keyboard.add_hotkey("F2", bridge.f2.emit)
        keyboard.add_hotkey("F3", bridge.f3.emit)
        keyboard.add_hotkey("esc", bridge.esc.emit)
    except Exception as e:
        print(f"热键注册失败（可能需要管理员权限）: {e}")
        return

    print(f"调试截图工具已启动，保存目录: {SAVE_DIR}")
    print("F2 全屏截图 | F3 框选区域 | ESC 取消框选 | Ctrl+C 退出")
    app.exec_()


def cancel_overlay(overlay_ref):
    overlay = overlay_ref["overlay"]
    if overlay is not None and overlay.isVisible():
        overlay.close()


if __name__ == "__main__":
    main()
