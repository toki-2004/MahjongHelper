import sys
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import mss
import keyboard

from vision import recognize_tiles_from_image, get_tile_name
from logic import decide_discard
from template_manager import update_template

CONFIG_FILE = "config.json"

# ------------------ 框选覆盖层 ------------------
class SelectionOverlay(QWidget):
    selection_done = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.showFullScreen()

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing = False

        self.mask_color = QColor(0, 0, 0, 180)
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

                info = f"{w} × {h}"
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


# ------------------ 结果覆盖层 ------------------
class ResultOverlay(QWidget):
    def __init__(self, screen_x, screen_y, region_w, region_h, helper=None):
        margin = 50
        self.margin = margin
        self.screen_x = screen_x
        self.screen_y = screen_y
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setGeometry(screen_x - margin, screen_y - margin,
                         region_w + 2*margin, region_h + 2*margin)

        self.tile_ids = []
        self.boxes = []
        self.background_color = QColor(0, 0, 0, 100)
        self.helper = helper

    def update_result(self, tile_ids, boxes):
        self.tile_ids = tile_ids
        self.boxes = boxes
        self.update()

    def paintEvent(self, event):
        if not self.tile_ids:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.background_color)

        for i, (bx, by, bw, bh) in enumerate(self.boxes):
            if i < len(self.tile_ids):
                rx = bx + self.margin
                ry = by + self.margin
                pen = QPen(QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.drawRect(rx, ry, bw, bh)

                name = get_tile_name(self.tile_ids[i])
                painter.setPen(QPen(Qt.white, 1))
                font = QFont("Arial", 12, QFont.Bold)
                painter.setFont(font)
                painter.drawText(rx + 2, ry - 5, name)

        if self.tile_ids:
            best_idx, score = decide_discard(self.tile_ids)
            if best_idx != -1:
                best_tile = get_tile_name(self.tile_ids[best_idx])
                text = f"建议打出：{best_tile}  (得分:{score})"
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 200))
                rect = QRect(self.width() - 300, 10, 280, 40)
                painter.drawRoundedRect(rect, 8, 8)
                painter.setPen(QPen(QColor(0, 255, 100), 1))
                font = QFont("Arial", 14, QFont.Bold)
                painter.setFont(font)
                painter.drawText(rect, Qt.AlignCenter, text)

    def mousePressEvent(self, event):
        # 只有交互模式启用时才处理右键纠错，否则忽略所有事件
        if event.button() == Qt.RightButton and self.helper and self.helper.interactive_enabled:
            pos = event.pos()
            for i, (bx, by, bw, bh) in enumerate(self.boxes):
                rx = bx + self.margin
                ry = by + self.margin
                if rx <= pos.x() <= rx + bw and ry <= pos.y() <= ry + bh:
                    self.correct_tile(i)
                    break
        else:
            # 非交互模式或非右键，忽略事件（实现透传）
            event.ignore()

    def correct_tile(self, index):
        if index >= len(self.tile_ids):
            return
        if self.helper is None:
            QMessageBox.information(self, "提示", "无法获取图像数据。")
            return
        gray_roi = self.helper.current_rois.get(index)
        if gray_roi is None:
            QMessageBox.information(self, "提示", "该牌的图像数据不存在，请重新识别。")
            return

        items = [f"{i} {get_tile_name(i)}" for i in range(34)]
        item, ok = QInputDialog.getItem(self, "修正模板", "选择正确的牌：", items, 0, False)
        if ok:
            correct_id = int(item.split()[0])
            success = update_template(correct_id, gray_roi, merge_ratio=0.3)
            if success:
                QMessageBox.information(self, "成功", "模板已更新！下次识别将生效。")
            else:
                QMessageBox.warning(self, "失败", "模板更新失败。")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


# ------------------ 设置对话框 ------------------
class SettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setFixedSize(350, 200)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("识别模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["手动 (F2 框选)", "自动 1s", "自动 2s", "自动 3s"])
        layout.addWidget(self.mode_combo)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.load_config()

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.mode_combo.setCurrentIndex(config.get('mode', 0))
        except:
            pass

    def save_config(self):
        config = {'mode': self.mode_combo.currentIndex()}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return config


# ------------------ 主UI窗口（左键托盘弹出） ------------------
class MainUI(QWidget):
    def __init__(self, helper):
        super().__init__()
        self.helper = helper
        self.setWindowTitle("麻将辅助 - 识别结果")
        self.setMinimumSize(400, 350)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()

        self.hand_label = QLabel("手牌：")
        self.hand_display = QTextEdit()
        self.hand_display.setReadOnly(True)
        self.hand_display.setMaximumHeight(100)

        self.suggestion_label = QLabel("建议：")
        self.suggestion_display = QTextEdit()
        self.suggestion_display.setReadOnly(True)
        self.suggestion_display.setMaximumHeight(60)

        self.status_label = QLabel("状态：等待识别")

        self.refresh_btn = QPushButton("刷新识别 (F1)")
        self.refresh_btn.clicked.connect(self.refresh)

        self.setting_btn = QPushButton("打开设置")
        self.setting_btn.clicked.connect(self.open_setting)

        # 交互开关
        self.interact_check = QCheckBox("覆盖层交互（右键纠错）")
        self.interact_check.setChecked(self.helper.interactive_enabled)
        self.interact_check.stateChanged.connect(self.toggle_interact)

        layout.addWidget(self.hand_label)
        layout.addWidget(self.hand_display)
        layout.addWidget(self.suggestion_label)
        layout.addWidget(self.suggestion_display)
        layout.addWidget(self.status_label)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.setting_btn)
        layout.addWidget(self.interact_check)

        self.setLayout(layout)
        self.update_display()

    def toggle_interact(self, state):
        enabled = (state == Qt.Checked)
        self.helper.set_overlay_interactive(enabled)

    def update_display(self):
        ids = self.helper.current_ids
        if ids:
            names = [get_tile_name(i) for i in ids]
            self.hand_display.setText(" ".join(names))
            best_idx, score = decide_discard(ids)
            if best_idx != -1:
                best_tile = get_tile_name(ids[best_idx])
                self.suggestion_display.setText(f"建议打出：{best_tile}  (得分:{score})")
            else:
                self.suggestion_display.setText("无法决策")
            self.status_label.setText(f"已识别 {len(ids)} 张牌")
        else:
            self.hand_display.setText("未识别到牌")
            self.suggestion_display.setText("")
            self.status_label.setText("暂无识别结果")

    def refresh(self):
        if self.helper.capture_region:
            self.helper.capture_and_recognize(self.helper.capture_region)
            self.update_display()
        else:
            QMessageBox.information(self, "提示", "请先框选区域 (F2)")

    def open_setting(self):
        self.helper.show_setting()

    def closeEvent(self, event):
        self.hide()
        event.ignore()


# ------------------ 主控制器 ------------------
class MahjongHelper(QObject):
    def __init__(self):
        super().__init__()
        self.sct = mss.MSS()
        self.overlay = None
        self.result_overlay = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_capture)
        self.config = self.load_config()
        self.is_auto = False
        self.capture_region = None
        self.current_ids = []
        self.current_boxes = []
        self.current_debug_img = None
        self.current_rois = {}
        self.main_ui = None

        # 默认开启交互
        self.interactive_enabled = True

        self.setup_tray()

        try:
            keyboard.add_hotkey('F2', lambda: QTimer.singleShot(0, self.start_selection))
            keyboard.add_hotkey('esc', lambda: QTimer.singleShot(0, self.cancel_selection))
            keyboard.add_hotkey('F1', lambda: QTimer.singleShot(0, self.refresh_recognition))
            print("全局热键注册成功：F2-框选，F1-刷新，ESC-取消。")
        except Exception as e:
            print(f"热键注册失败: {e}，请以管理员身份运行。")

        # 启动时不进入框选模式

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'mode': 0}

    def setup_tray(self):
        self.tray = QSystemTrayIcon()
        icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        self.tray.setIcon(icon)
        self.tray.setToolTip("麻将辅助工具")
        self.tray.activated.connect(self.on_tray_activated)

        self.tray_menu = QMenu()
        action_setting = QAction("设置", self)
        action_setting.triggered.connect(self.show_setting)
        self.tray_menu.addAction(action_setting)

        action_capture = QAction("框选区域 (F2)", self)
        action_capture.triggered.connect(self.start_selection)
        self.tray_menu.addAction(action_capture)

        action_refresh = QAction("刷新识别 (F1)", self)
        action_refresh.triggered.connect(self.refresh_recognition)
        self.tray_menu.addAction(action_refresh)

        self.tray_menu.addSeparator()
        action_exit = QAction("退出", self)
        action_exit.triggered.connect(self.exit_app)
        self.tray_menu.addAction(action_exit)

        self.tray.setContextMenu(self.tray_menu)
        self.tray.show()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_main_ui()

    def show_main_ui(self):
        if self.main_ui is None:
            self.main_ui = MainUI(self)
        if self.main_ui.isVisible():
            self.main_ui.activateWindow()
            self.main_ui.raise_()
        else:
            self.main_ui.show()
            self.main_ui.update_display()
            # 同步复选框状态
            self.main_ui.interact_check.setChecked(self.interactive_enabled)

    def show_setting(self):
        dialog = SettingDialog()
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.save_config()
            self.config = config
            mode = config.get('mode', 0)
            if mode > 0:
                interval = [0, 1000, 2000, 3000][mode]
                self.start_auto(interval)
            else:
                self.stop_auto()

    def start_auto(self, interval):
        self.is_auto = True
        self.timer.start(interval)
        self.tray.showMessage("麻将辅助", f"已启动自动识别，间隔 {interval}ms")

    def stop_auto(self):
        self.is_auto = False
        self.timer.stop()

    def auto_capture(self):
        if self.capture_region:
            self.capture_and_recognize(self.capture_region)

    def start_selection(self):
        if self.result_overlay and self.result_overlay.isVisible():
            self.result_overlay.hide()
        if self.overlay:
            self.overlay.close()
            self.overlay.deleteLater()
            self.overlay = None
        self.overlay = SelectionOverlay()
        self.overlay.selection_done.connect(self.on_selection_done)
        self.overlay.show()

    def cancel_selection(self):
        if self.overlay and self.overlay.isVisible():
            self.overlay.close()

    def refresh_recognition(self):
        if self.capture_region:
            self.capture_and_recognize(self.capture_region)
            if self.main_ui and self.main_ui.isVisible():
                self.main_ui.update_display()
        else:
            self.start_selection()

    def on_selection_done(self, x, y, w, h):
        if x == 0 and y == 0 and w == 0 and h == 0:
            return
        if self.overlay:
            self.overlay.close()
            self.overlay = None

        self.capture_region = (x, y, w, h)
        if self.result_overlay:
            self.result_overlay.close()
            self.result_overlay.deleteLater()
            self.result_overlay = None

        self.capture_and_recognize((x, y, w, h))
        if self.is_auto:
            self.timer.start(self.timer.interval())

    def set_overlay_interactive(self, enabled):
        self.interactive_enabled = enabled
        if self.result_overlay:
            if enabled:
                # 启用交互：移除透传属性，窗口可接收鼠标事件
                self.result_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            else:
                # 禁用交互：设置透传属性，所有鼠标事件穿透
                self.result_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            # 强制刷新
            self.result_overlay.setWindowFlags(self.result_overlay.windowFlags())
            self.result_overlay.show()
        if self.main_ui and self.main_ui.isVisible():
            self.main_ui.interact_check.setChecked(enabled)

    def capture_and_recognize(self, region):
        # 先隐藏覆盖层，避免干扰截图
        if self.result_overlay and self.result_overlay.isVisible():
            self.result_overlay.hide()

        try:
            x, y, w, h = region
            monitor = {"left": x, "top": y, "width": w, "height": h}
            img = self.sct.grab(monitor)
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

            ids, boxes, debug_img = recognize_tiles_from_image(img_bgr)

            self.current_ids = ids
            self.current_boxes = boxes
            self.current_debug_img = debug_img

            # 存储 ROI
            self.current_rois = {}
            for i, (bx, by, bw, bh) in enumerate(boxes):
                if i < len(ids):
                    roi = img_bgr[by:by+bh, bx:bx+bw]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        self.current_rois[i] = gray_roi

            self.show_result_overlay(x, y, w, h, ids, boxes)

            if self.main_ui and self.main_ui.isVisible():
                self.main_ui.update_display()

        except Exception as e:
            print(f"识别错误: {e}")
            self.tray.showMessage("识别错误", str(e))
            if self.result_overlay and not self.result_overlay.isVisible():
                self.result_overlay.show()

    def show_result_overlay(self, screen_x, screen_y, region_w, region_h, ids, boxes):
        if self.result_overlay is None:
            self.result_overlay = ResultOverlay(screen_x, screen_y, region_w, region_h, helper=self)
        else:
            self.result_overlay.setGeometry(screen_x - 50, screen_y - 50,
                                            region_w + 100, region_h + 100)
            self.result_overlay.screen_x = screen_x
            self.result_overlay.screen_y = screen_y
            self.result_overlay.margin = 50
        # 应用当前交互模式
        self.set_overlay_interactive(self.interactive_enabled)
        self.result_overlay.show()
        self.result_overlay.update_result(ids, boxes)

    def exit_app(self):
        if self.timer:
            self.timer.stop()
        QApplication.quit()


if __name__ == "__main__":
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("建议以管理员身份运行，以确保全局热键正常工作。")
    except:
        pass

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    helper = MahjongHelper()
    sys.exit(app.exec_())