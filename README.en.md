# 🀄 MahjongHelper

> **Language:** English | [简体中文](README.md)

A mahjong assistant based on real-time screen recognition: select the hand-tile area on the screen, and it automatically recognizes the tiles and suggests the best discard. Supports manual selection and scheduled auto-refresh; templates are built with the companion capture tool (template_capture.py), and recognition accuracy improves gradually as samples accumulate.

![GitHub release (latest by date)](https://img.shields.io/github/v/release/toki-2004/MahjongHelper)
![GitHub](https://img.shields.io/github/license/toki-2004/MahjongHelper)

---

## ✨ Features

- **Real-time recognition**: captures a specified screen region and automatically recognizes each tile's suit and rank (template matching + Viterbi sequence smoothing).
- **Smart discard suggestions**: computes Japanese mahjong shanten (standard forms / seven pairs / thirteen orphans) plus an evaluation of useful draws, and recommends the tile whose discard leaves the lowest shanten and the most accepted tiles; red 5s are kept preferentially as dora tiles; with 13 tiles it shows the current shanten and the tile to draw. The suggestion text is displayed as a single translucent label above the recognition region.
- **Green highlight**: the name label of the recommended discard is shown in **green** — clear and intuitive.
- **Template capture tool**: builds a 37-class template library (red 5s included) through semi-automatic annotation of real screenshots; the more samples, the better the recognition.
- **Raw-pixel matching**: online mahjong scenes render stably, so both templates and live recognition use raw-pixel matching with no preprocessing such as denoising/whitening, preserving every discriminative detail.
- **Scheduled refresh**: automatic recognition can be enabled in the settings (1s / 2s / 3s intervals).
- **Global hotkeys + system tray**: lives in the tray without interrupting your workflow.
- **Debug capture tool**: ships with `debug_capture.py`, which can capture the full screen or a selected region in one click and save test images, making it easy to collect samples of all kinds of tile faces.
- **Unzip and run**: the Release provides a complete archive; no Python environment needs to be installed.

---

## 📥 Download and Usage

### Option 1: Download the Release archive (recommended)

1. Go to the [Releases page](https://github.com/toki-2004/MahjongHelper/releases) and download the latest archive (e.g. `MahjongHelper-v1.1.0.zip`).
2. Extract it to any directory. Keep `MahjongHelper.exe` and the `_internal` folder in the same directory, and **do not move the exe on its own**.
3. Run `MahjongHelper.exe`. If the game runs as administrator, start this tool as administrator as well so that the global hotkeys can capture key presses.
4. After launch, press **F2** to select the hand-tile region; the program recognizes it automatically and displays the result.

> Tip: `config.json` (user settings), `mahjong_helper.log` (log), and `debug_output` (debug output) are generated automatically in the directory where the exe is located.

### Option 2: Run from Source

Requires Python 3.8+ (3.10+ recommended).

```bash
git clone https://github.com/toki-2004/MahjongHelper.git
cd MahjongHelper
pip install -r requirements.txt
python main.py
```

> The source version uses `templates/` in the project root as tile templates; the packaged version has them built in at `_internal/templates`.

---

## ⌨️ Hotkeys

| Key | Function |
|------|------|
| `F2` | Enter selection mode (redefine the recognition region) |
| `F1` | Refresh recognition (reuse the current region and recognize again) |
| `ESC` | Cancel the current operation (cancel the selection or close the overlay) |

---

## 🧩 Templates and Customization

- **Selection tips**: after pressing F2, drag the mouse; it is best to select only one complete row of tiles to reduce interference from the background and other elements, improving recognition stability.
- **Auto mode**: right-click the tray icon → Settings to switch to "Auto 1s / 2s / 3s"; the program then recognizes the current region on a timer.
- **Template directories**: `templates/` for the source version, `_internal/templates/` for the packaged version. Templates and samples are all stored as raw pixels with no preprocessing during recognition, keeping the details of online mahjong scenes intact.
- **Rebuilding templates**: use `template_capture.py` to automatically detect tiles in screenshots, manually annotate each tile's true suit, and save samples; after collecting, click "Rebuild template library" to generate the templates. The main recognition UI provides no online-correction dropdown.

---

## 🀄 Template Capture and Rebuild (template_capture.py)

Used to build a template library in bulk from real screenshots, without requiring any existing templates: the program first locates each tile's position and count automatically through geometric features (white tile faces, size, spacing), then each tile's true suit is annotated manually; the results are saved as samples and used to rebuild the templates.

```bash
python template_capture.py
```

Workflow:

1. Click "Open Image" and choose a hand-tile screenshot (preferably one complete row of tiles captured beforehand with `debug_capture.py`, or a game screenshot).
2. Click "Detect Tiles"; the program detects and numbers every tile automatically (only the main row is kept; side tiles from chi/pon/kan calls are filtered out; supports 14, 13, 11, 10, 8, 7, 5, 4, 2, and 1 tiles).
3. In the dropdowns on the right, select each tile's true suit; choose "(skip)" for tiles you don't need.
4. Click "Save This Image's Samples"; samples are written to `templates/samples/{牌id}/` (sizes are unified automatically, raw pixels preserved).
5. Repeat with more screenshots until all 37 tile classes are covered (34 regular tiles + 3 red 5s), then click "Rebuild Template Library".

Rebuild rule: for each class, one **real representative sample** is chosen from all samples (after translation alignment, the one closest to the class median); no blending/averaging is performed, avoiding templates becoming blurrier as they accumulate; the raw samples are kept as well for sample-level matching.

Tip: collect 5~10 samples per tile class from different game scenes (different brightness, zoom, and tile orientations) for a more robust template library.

## 🔨 Debug Capture Tool

`debug_capture.py` comes with a graphical interface for quickly collecting test screenshots of all kinds of tile faces: the capture hotkey and save directory are customizable, and settings are stored in `capture_config.json`, taking effect automatically after restart. The window lives in the system tray; closing the window does not exit the program.

```bash
python debug_capture.py [monitor_index]
```

| Key | Function |
|------|------|
| `F2` (changeable in the UI) | Capture the screen under the mouse and save |
| `F3` (changeable in the UI) | Drag a region selection on the screen under the mouse and save |
| `ESC` | Cancel the selection |

> Without arguments it automatically captures the screen under the mouse (true for both F2 and F3); you can also pass 1, 2, 3… to specify a screen by its mss monitor index (each screen's index is listed in the log at startup). Images are saved to the specified directory with file names auto-incremented from existing ones (e.g. if `1.png` exists, the next is saved as `2.png`), ready to be used directly for self-testing recognition or template training.

---

## ❓ FAQ

- **Hotkeys not responding**: make sure the program is running and has not been minimized to the tray; if the game runs as administrator, this program must be started with the same privileges; if it is blocked by antivirus software, add the program to the whitelist.
- **0 tiles recognized or `?` displayed**: the selected region is too small or the tiles are rendered too small — it is recommended to zoom the image/window to 100% or more before recognizing; `?` means that slot could not be recognized, so re-select a more complete tile row and press F1 to refresh.
- **The recognized count is not 13~14 tiles**: when selecting, try to include only one row of tiles, avoiding capturing adjacent tiles, the background, or other areas.
- **How discard suggestions work**: for every candidate tile, the program computes the shanten and useful draws after discarding it (tiles that lower the shanten when drawn, with remaining counts computed as 4 minus the tiles held in hand), preferring lower shanten and more useful draws; red 5s are kept when all else is equal; when tenpai, the winning tiles are shown.

---

## 📦 Building It Yourself (Developers)

The repository ships with a PyInstaller configuration (one-directory mode, UPX disabled; startup speed comparable to running from source):

```bash
pip install pyinstaller
python -m PyInstaller --noconfirm --clean MahjongHelper.spec
```

The output is located in `dist/MahjongHelper/`; compress the **entire directory** and it is ready to distribute.

---

## 📁 Project Structure

```
main.py              主程序：界面、热键、识别线程
vision.py            图像识别：区域提取、分档对齐、模板匹配
template_manager.py  模板加载、矩阵预计算、样本管理、模板重建
logic.py             日麻切牌建议（向听数 + 有效进张 + 红5 保留）
debug_capture.py     截图工具 GUI：自定义热键与保存目录，F2 全屏 / F3 框选保存
templates/           37 张牌面模板（0.png ~ 36.png，含 34~36 红5万/索/筒）
template_packs/      模板包仓库：按游戏/皮肤组织（如 雀魂默认/templates），供不同牌面皮肤切换
MahjongHelper.spec   PyInstaller 打包配置
```

---

## 📄 License

This project is released under the **MIT License**; see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Thanks to the excellent libraries from the open-source community: PyQt5, OpenCV, NumPy, Pillow, mss, keyboard, and more.

---
