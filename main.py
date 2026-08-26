import sys
import os
import json
import logging
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import mss
import keyboard

# ========== 日志配置 ==========
LOG_FILE = os.path.join(os.getcwd(), "mahjong_helper.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 资源路径适配 ==========
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

CONFIG_FILE = os.path.join(os.getcwd(), "config.json")
TEMPLATE_PATH = resource_path("templates")

# 修改 template_manager 的模板路径
import template_manager
template_manager.TEMPLATE_PATH = TEMPLATE_PATH
from template_manager import load_templates, templates, get_tile_name
logger.info(f"模板路径: {TEMPLATE_PATH}")
templates = load_templates()
logger.info(f"模板加载完成，共 {len(templates)} 张")

from vision import recognize_tiles_from_image
from logic import decide_discard


# ================== 识别线程 ==================
class RecognizeThread(QThread):
    result_ready = pyqtSignal(list, list, np.ndarray)  # ids, boxes, debug_img

    def __init__(self, region):
        super().__init__()
        self.region = region

    def run(self):
        logger.info("识别线程启动")
        start_time = time.time()
        try:
            sct = mss.MSS()
            x, y, w, h = self.region
            monitor = {"left": x, "top": y, "width": w, "height": h}
            img = sct.grab(monitor)
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            logger.info(f"截图完成，尺寸: {img_bgr.shape}")

            ids, boxes, debug_img = recognize_tiles_from_image(img_bgr)
            logger.info(f"识别完成，得到 {len(ids)} 张牌")

            elapsed = time.time() - start_time
            logger.info(f"识别总耗时: {elapsed:.2f}秒")
            self.result_ready.emit(ids, boxes, debug_img)

        except Exception as e:
            logger.error(f"识别错误: {e}", exc_info=True)
            self.result_ready.emit([], [], None)


# ================== 界面类 ==================
class SelectionOverlay(QWidget):
    selection_done = pyqtSignal(int, int, int, int)

    def __init__(self, screen=None):
        super().__init__(None)
        self._screen = screen
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        if screen is not None:
            g = screen.geometry()
            self.setGeometry(g.x(), g.y(), g.width(), g.height())
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
                g = self._screen.geometry()
                # 转成虚拟桌面绝对坐标，mss 按绝对坐标抓取，多屏才不会错位
                self.selection_done.emit(g.x() + x, g.y() + y, w, h)
            else:
                self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.selection_done.emit(0, 0, 0, 0)


class ResultOverlay(QWidget):
    def __init__(self, screen_x, screen_y, region_w, region_h, helper=None):
        margin = 20
        self.margin = margin
        self.screen_x = screen_x
        self.screen_y = screen_y
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setMouseTracking(True)
        self.setGeometry(screen_x - margin, screen_y - margin,
                         region_w + 2*margin, region_h + 2*margin)

        self.tile_ids = []
        self.boxes = []
        self.helper = helper

        self.label_list = []
        self.suggestion_label = None

    def update_result(self, tile_ids, boxes):
        for lbl in self.label_list:
            lbl.deleteLater()
        self.label_list.clear()
        if self.suggestion_label is not None:
            self.suggestion_label.deleteLater()
            self.suggestion_label = None

        self.tile_ids = tile_ids
        self.boxes = boxes

        best_idx, _ = decide_discard(tile_ids) if tile_ids else (-1, -1)

        for i, (bx, by, bw, bh) in enumerate(boxes):
            if i >= len(tile_ids):
                break

            lbl = QLabel(self)
            lbl.setText(get_tile_name(tile_ids[i]))
            if i == best_idx:
                lbl.setStyleSheet("color: green; background-color: rgba(0,0,0,150); font-weight: bold; padding: 2px;")
            else:
                lbl.setStyleSheet("color: white; background-color: rgba(0,0,0,150); font-weight: bold; padding: 2px;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl_x = bx + self.margin
            lbl_y = by + self.margin + bh + 2
            lbl_w = bw
            lbl_h = 20
            lbl.setGeometry(lbl_x, lbl_y, lbl_w, lbl_h)
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lbl.show()
            self.label_list.append(lbl)

        # 建议打出：半透明标签贴在识别区域上方，横跨整个区域宽度
        if best_idx >= 0 and best_idx < len(tile_ids):
            sug = QLabel(self)
            sug.setText("建议打出：%s" % get_tile_name(tile_ids[best_idx]))
            sug.setStyleSheet(
                "color: yellow; background-color: rgba(0,0,0,150);"
                "font-weight: bold; padding: 2px;")
            sug.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            sug.setWordWrap(False)
            sug.setGeometry(self.margin, self.margin - 24,
                            self.width() - 2 * self.margin, 22)
            sug.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            sug.show()
            self.suggestion_label = sug

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


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


class MainUI(QWidget):
    def __init__(self, helper):
        super().__init__()
        self.helper = helper
        self.setWindowTitle("麻将辅助 - 识别结果")
        self.setMinimumSize(400, 350)

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

        layout.addWidget(self.hand_label)
        layout.addWidget(self.hand_display)
        layout.addWidget(self.suggestion_label)
        layout.addWidget(self.suggestion_display)
        layout.addWidget(self.status_label)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.setting_btn)

        self.setLayout(layout)
        self.update_display()

    def update_display(self):
        ids = self.helper.current_ids
        if ids:
            names = [get_tile_name(i) for i in ids]
            self.hand_display.setText(" ".join(names))
            best_idx, info = decide_discard(ids)
            if best_idx != -1:
                best_tile = get_tile_name(ids[best_idx])
                if info:
                    eff_names = " ".join("%s×%d" % (n, r) for n, r in info['effective_tiles'][:8])
                    self.suggestion_display.setText(
                        f"建议打出：{best_tile}（打出后 {info['shanten']} 向听，"
                        f"有效进张 {info['effective']} 张）\n{eff_names}")
                else:
                    self.suggestion_display.setText(f"建议打出：{best_tile}")
            elif info:
                eff_names = " ".join("%s×%d" % (n, r) for n, r in info['effective_tiles'][:8])
                self.suggestion_display.setText(
                    f"当前 {info['shanten']} 向听，有效进张 {info['effective']} 张\n{eff_names}")
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


class MahjongHelper(QObject):
    # 全局热键信号：keyboard 回调在独立线程，通过信号队列投递到主线程
    hotkey_f2 = pyqtSignal()
    hotkey_f1 = pyqtSignal()
    hotkey_esc = pyqtSignal()

    def __init__(self):
        super().__init__()
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
        self.main_ui = None
        self.rec_thread = None

        self.setup_tray()

        try:
            self.hotkey_f2.connect(self.start_selection)
            self.hotkey_f1.connect(self.refresh_recognition)
            self.hotkey_esc.connect(self.cancel_selection)
            keyboard.add_hotkey('F2', self.hotkey_f2.emit)
            keyboard.add_hotkey('F1', self.hotkey_f1.emit)
            keyboard.add_hotkey('esc', self.hotkey_esc.emit)
            logger.info("全局热键注册成功：F2-框选，F1-刷新，ESC-取消。")
        except Exception as e:
            logger.error(f"热键注册失败: {e}，请以管理员身份运行。")

        QTimer.singleShot(200, self.show_main_ui)

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
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        self.overlay = SelectionOverlay(screen)
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

    def capture_and_recognize(self, region):
        if self.result_overlay and self.result_overlay.isVisible():
            self.result_overlay.hide()
        logger.info(f"开始识别区域: {region}")
        if self.rec_thread and self.rec_thread.isRunning():
            logger.info("上一轮识别尚未完成，跳过本次")
            return
        self.rec_thread = RecognizeThread(region)
        self.rec_thread.result_ready.connect(self.on_recognition_done)
        self.rec_thread.start()

    def on_recognition_done(self, ids, boxes, debug_img):
        logger.info(f"收到识别结果，牌数: {len(ids)}")
        self.current_ids = ids
        self.current_boxes = boxes
        self.current_debug_img = debug_img

        if self.capture_region:
            x, y, w, h = self.capture_region
            self.show_result_overlay(x, y, w, h, ids, boxes)

        if self.main_ui and self.main_ui.isVisible():
            self.main_ui.update_display()

    def show_result_overlay(self, screen_x, screen_y, region_w, region_h, ids, boxes):
        if self.result_overlay is None:
            self.result_overlay = ResultOverlay(screen_x, screen_y, region_w, region_h, helper=self)
        else:
            self.result_overlay.setGeometry(screen_x - 20, screen_y - 20,
                                            region_w + 40, region_h + 40)
            self.result_overlay.screen_x = screen_x
            self.result_overlay.screen_y = screen_y
            self.result_overlay.margin = 20
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
