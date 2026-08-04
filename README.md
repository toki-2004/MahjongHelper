# 🀄️ MahjongHelper

**基于屏幕实时识别的麻将辅助工具**，通过截取牌面图像自动分析手牌，给出最优切牌建议。支持手动框选、自动刷新、模板在线修正，帮助您更快决策。

![GitHub release (latest by date)](https://img.shields.io/github/v/release/toki-2004/MahjongHelper)
![GitHub](https://img.shields.io/github/license/toki-2004/MahjongHelper)

---

## ✨ 功能特点

- **实时识别**：截取屏幕任意区域的麻将牌，自动识别花色与点数。
- **智能建议**：基于“关联度”算法，计算打出哪张牌能使剩余牌组成更多顺子、刻子或对子，推荐最优切牌。
- **绿色高亮**：被建议打出的牌，其名称标签会显示为**绿色**，一目了然。
- **修正机制**：若识别有误，可通过牌下方的下拉栏选择正确牌，程序自动更新模板并刷新识别，越用越准。
- **热键操作**：F2 框选区域，F1 刷新识别，ESC 取消。
- **系统托盘**：常驻托盘，左键呼出主界面，右键快捷操作。
- **独立发布包**：提供可直接运行的 `.exe` 文件，无需安装 Python 环境。

---

## 📥 下载与使用

### 方式一：直接运行 EXE（推荐）

前往 [Releases 页面](https://github.com/toki-2004/MahjongHelper/releases) 下载最新版本的 `MahjongHelper.exe`。  
下载后，**以管理员身份运行**（全局热键需要管理员权限）。

启动后：
1. 按 **F2** 进入框选模式，拖拽鼠标框出你的手牌区域。
2. 释放鼠标，程序自动识别并显示结果。
3. 按 **F1** 可随时刷新识别（牌面变化时使用）。
4. 若某张牌识别错误，点击其下方的下拉栏，选择正确牌，模板会自动更新。

### 方式二：从源码运行

需要 Python 3.8+ 环境。

```bash
git clone https://github.com/toki-2004/MahjongHelper.git
cd MahjongHelper
pip install -r requirements.txt
python main.py
```

> **提示**：`templates/` 目录需要包含 34 张模板图片（`0.png`~`33.png`），否则识别会失败。如果你没有模板，可以使用程序自带的占位图，但建议自行准备与游戏风格匹配的牌面图片。

---

## ⌨️ 快捷键

| 按键 | 功能 |
|------|------|
| `F2` | 进入框选模式（重新划定识别区域） |
| `F1` | 刷新识别（不改变区域，重新识别当前画面） |
| `ESC` | 取消当前操作（取消框选或关闭覆盖层） |

---

## 🧩 自定义与修正

- **框选区域**：按 F2 后拖拽鼠标，只选包含牌面的区域，减少背景干扰。
- **修正识别**：每张牌下方都有一个下拉栏，列出了所有 34 种牌。选择正确的牌后，程序会将该牌的图像与模板融合，下次识别更准。
- **自动模式**：右键托盘图标 → 设置，可切换为“自动 1s/2s/3s”，程序会定时自动识别当前区域。

---

## 📦 打包为 EXE（仅开发者）

如需自行打包：

1. 确保已安装 `pyinstaller`。
2. 在项目根目录执行：

```bash
pyinstaller --onefile --windowed --name MahjongHelper --add-data "templates;templates" main.py
```

3. 生成的 `MahjongHelper.exe` 位于 `dist/` 目录。

> 注意：`config.json` 会在 exe 同级目录生成，用于保存用户设置；`templates` 文件夹已打包进 exe，无需额外放置。

---

## 📄 许可证

本项目采用 **MIT License**，详情请见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

本项目得益于开源社区众多优秀库，包括 PyQt5、OpenCV、mss、keyboard 等。

---

**Enjoy your game!** 🀄️🎴