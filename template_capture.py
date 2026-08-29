# -*- coding: utf-8 -*-
"""Template capture GUI.

Load a screenshot, auto-detect individual tiles (geometry only, no template
needed), label each tile with its real suit/value by hand, then save the ROI
as a sample. Rebuild the template library afterwards - samples are kept as-is
and the per-class template is a real representative sample, never a blurry
average.

Usage:
    python template_capture.py
"""

import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

import template_manager as tm
from template_manager import TEMPLATE_SIZE, get_tile_name
from tile_detect import VALID_TILE_COUNTS, detect_tile_boxes

SAMPLE_ROOT = os.path.join("templates", "samples")
TILE_NAMES = [get_tile_name(i) for i in range(tm.TOTAL_TILE_TYPES)]


def _process_roi(roi):
    """Resize ROI to template size only. Raw pixels are kept so that the
    stable electronic-mahjong rendering is matched without information loss."""
    h, w = TEMPLATE_SIZE
    return cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)


def _next_sample_path(tile_id):
    d = os.path.join(SAMPLE_ROOT, str(tile_id))
    os.makedirs(d, exist_ok=True)
    existing = [f for f in os.listdir(d) if f.endswith(".png")]
    n = len(existing) + 1
    return os.path.join(d, "sample_%03d.png" % n)


class CaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("麻将模板采集工具")
        self.resize(1280, 760)

        self.img_bgr = None
        self.img_raw = None
        self.img_path = None
        self.boxes = []
        self.combos = []
        self.file_queue = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- top buttons ----
        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_detect = QPushButton("检测牌")
        self.btn_fill = QPushButton("用模板识别填充")
        self.btn_save = QPushButton("保存本图样本")
        self.btn_next = QPushButton("下一张")
        self.btn_rebuild = QPushButton("重建模板库")
        self.btn_open.clicked.connect(self.open_images)
        self.btn_detect.clicked.connect(self.detect)
        self.btn_fill.clicked.connect(self.auto_fill)
        self.btn_save.clicked.connect(self.save_samples)
        self.btn_next.clicked.connect(self.load_next)
        self.btn_rebuild.clicked.connect(self.rebuild)
        for b in (self.btn_open, self.btn_detect, self.btn_fill, self.btn_save,
                  self.btn_next, self.btn_rebuild):
            bar.addWidget(b)
        root.addLayout(bar)

        # ---- image + labels ----
        mid = QHBoxLayout()
        self.img_label = QLabel("请先打开截图")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(760, 500)
        self.img_label.setStyleSheet("border: 1px solid #888; background: #222;")
        mid.addWidget(self.img_label, 3)

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.list_widget)
        scroll.setMinimumWidth(340)
        mid.addWidget(scroll, 1)
        root.addLayout(mid, 1)

        self.status = QLabel("就绪")
        root.addWidget(self.status)
        self._clear_list()
        self._update_status()

    # ---------- helpers ----------
    def _clear_list(self):
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.combos = []

    def _update_status(self, extra=""):
        if self.img_path:
            base = os.path.basename(self.img_path)
        else:
            base = "无图片"
        counts = []
        if os.path.isdir(SAMPLE_ROOT):
            for d in sorted(os.listdir(SAMPLE_ROOT)):
                if d.isdigit():
                    p = os.path.join(SAMPLE_ROOT, d)
                    counts.append("%s=%d" % (d, len([f for f in os.listdir(p) if f.endswith('.png')])))
        text = "%s | %s | 已采集: %s" % (base, extra, " ".join(counts) if counts else "无")
        self.status.setText(text)

    def _show_image(self):
        if self.img_bgr is None:
            return
        h, w = self.img_bgr.shape[:2]
        label_w = max(self.img_label.width(), 1)
        label_h = max(self.img_label.height(), 1)
        scale = min(label_w / float(w), label_h / float(h), 2.0)
        disp = cv2.resize(self.img_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                      QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

    def _draw_boxes(self):
        if self.img_bgr is None:
            return
        img = self.img_bgr.copy()
        for i, (bx, by, bw, bh) in enumerate(self.boxes):
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(img, str(i + 1), (bx + 2, by + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.img_bgr = img

    def auto_fill(self):
        """Use the current template/sample library to recognize every detected
        tile and pre-select its combo. Low-confidence or ambiguous matches are
        left at "(跳过)" so the user only has to fix the uncertain ones."""
        if not self.boxes or not self.combos or self.img_raw is None:
            QMessageBox.information(self, "提示", "请先检测牌。")
            return
        from vision import match_template_with_scores_exemplar
        classes = len(set(tm.EXEMPLAR_IDS)) if tm.EXEMPLAR_IDS else 0
        if classes < 3:
            self._update_status("模板样本不足（当前仅 %d 类），请先手动标注至少 3 类后自动填充才可用" % classes)
            return
        filled = 0
        h, w = TEMPLATE_SIZE
        for i, combo in enumerate(self.combos):
            bx, by, bw, bh = self.boxes[i]
            roi = self.img_raw[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                continue
            roi = _process_roi(roi)
            scores = match_template_with_scores_exemplar(roi)
            if not scores:
                continue
            best_id, best_conf = scores[0]
            second_conf = scores[1][1] if len(scores) > 1 else -1.0
            if best_id >= 0 and best_conf >= 0.45 and (best_conf - second_conf) >= 0.03:
                combo.setCurrentIndex(best_id + 1)
                filled += 1
        self._update_status("自动填充 %d/%d 张（低置信度保留待人工确认）" % (
            filled, len(self.combos)))

    # ---------- actions ----------
    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择截图", os.getcwd(), "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        self.file_queue = list(files)
        self.load_next()

    def load_next(self):
        if not self.file_queue:
            return
        path = self.file_queue.pop(0)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "读取失败", "无法读取: %s" % path)
            self.load_next()
            return
        self.img_path = path
        self.img_raw = img
        self.img_bgr = img
        self.boxes = []
        self._clear_list()
        self._show_image()
        self._update_status("已加载，请点击“检测牌”")

    def detect(self):
        if self.img_bgr is None:
            QMessageBox.information(self, "提示", "请先打开截图。")
            return
        img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        if img is None:
            return
        self.img_raw = img
        self.img_bgr = img
        self.boxes, count = detect_tile_boxes(img)
        if not self.boxes:
            QMessageBox.warning(self, "未检测到牌", "没有找到牌面，请换一张截图或调整框选。")
            self._update_status("未检测到牌")
            return
        self._draw_boxes()
        self._show_image()
        self._clear_list()
        for i in range(len(self.boxes)):
            row = QHBoxLayout()
            idx = QLabel("#%d" % (i + 1))
            idx.setMinimumWidth(40)
            combo = QComboBox()
            combo.addItem("(跳过)")
            combo.addItems(TILE_NAMES)
            combo.setMinimumWidth(150)
            row.addWidget(idx)
            row.addWidget(combo, 1)
            self.list_layout.addLayout(row)
            self.combos.append(combo)
        warn = ""
        if count not in VALID_TILE_COUNTS:
            warn = "（注意：检测到 %d 张，不在常规牌数内，请检查框选区域）" % count
        self._update_status("检测到 %d 张牌%s" % (count, warn))
        self.auto_fill()

    def save_samples(self):
        if not self.boxes or not self.combos:
            QMessageBox.information(self, "提示", "请先检测牌。")
            return
        saved = []
        skipped = 0
        for i, combo in enumerate(self.combos):
            sel = combo.currentIndex()
            if sel <= 0:
                skipped += 1
                continue
            tile_id = sel - 1
            bx, by, bw, bh = self.boxes[i]
            if bw <= 0 or bh <= 0 or by + bh > self.img_bgr.shape[0] or bx + bw > self.img_bgr.shape[1]:
                skipped += 1
                continue
            raw = self.img_raw if self.img_raw is not None else self.img_bgr
            roi = raw[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                skipped += 1
                continue
            path = _next_sample_path(tile_id)
            # 同上：中文路径下 imwrite 会静默失败
            ok, buf = cv2.imencode(".png", _process_roi(roi))
            if not ok:
                skipped += 1
                continue
            buf.tofile(path)
            saved.append((tile_id, path))
        if saved:
            QMessageBox.information(
                self, "保存完成",
                "已保存 %d 张样本（跳过 %d 张）\n%s" % (
                    len(saved), skipped,
                    "\n".join("%s -> %s" % (get_tile_name(t), p) for t, p in saved)))
        else:
            QMessageBox.information(self, "未保存", "没有选择任何牌（全部跳过）。")
        tm.load_samples()  # 新样本立即参与后续识别填充
        self._update_status("已保存 %d 张样本" % len(saved))

    def rebuild(self):
        built = tm.rebuild_templates_from_samples()
        if not built:
            QMessageBox.information(self, "无样本", "samples 目录为空，请先采集样本。")
            return
        tm.load_templates()
        tm.load_samples()
        missing = [i for i in range(tm.TOTAL_TILE_TYPES) if i not in {t for t, _ in built}]
        lines = ["已重建 %d 类模板（样本数）：" % len(built)]
        for tid, n in sorted(built):
            lines.append("  %s : %d 样本" % (get_tile_name(tid), n))
        if missing:
            lines.append("仍缺 %d 类: %s" % (
                len(missing), " ".join(get_tile_name(i) for i in missing)))
        QMessageBox.information(self, "重建完成", "\n".join(lines))
        self._update_status("模板库已重建（%d 类）" % len(built))


def main():
    app = QApplication(sys.argv)
    win = CaptureWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
