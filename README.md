# 🀄️ MahjongHelper

基于屏幕实时识别的麻将辅助工具：框选屏幕中的手牌区域，自动识别牌面并给出最优切牌建议。支持手动框选、定时自动刷新、下拉框在线修正模板，越用越准。

![GitHub release (latest by date)](https://img.shields.io/github/v/release/toki-2004/MahjongHelper)
![GitHub](https://img.shields.io/github/license/toki-2004/MahjongHelper)

---

## ✨ 功能特点

- **实时识别**：截取屏幕指定区域，自动识别牌的花色与点数（模板匹配 + 维特比序列平滑）。
- **智能切牌建议**：基于牌型组合权重（顺子、刻子、对子、搭子）计算打出每张候选牌后剩余牌的得分，推荐最优切牌。
- **绿色高亮**：被建议打出的牌，其名称标签显示为**绿色**，一目了然。
- **在线修正**：识别有误时，通过牌下方的下拉栏选择正确牌，程序自动用该牌图像替换对应模板并**立即生效**。
- **定时刷新**：可在设置中开启自动识别（1s / 2s / 3s 间隔）。
- **全局热键 + 系统托盘**：常驻托盘，不打断操作。
- **解压即用**：Release 提供完整压缩包，无需安装 Python 环境。

---

## 📥 下载与使用

### 方式一：下载 Release 压缩包（推荐）

1. 前往 [Releases 页面](https://github.com/toki-2004/MahjongHelper/releases) 下载最新版本的压缩包（如 `MahjongHelper-v1.1.0.zip`）。
2. 解压到任意目录。请保持 `MahjongHelper.exe` 与 `_internal` 文件夹在同一目录下，**不要单独移动 exe**。
3. 右键 **以管理员身份运行** `MahjongHelper.exe`（全局热键需要管理员权限）。
4. 启动后按 **F2** 框选手牌区域，程序自动识别并显示结果。

> 提示：`config.json`（用户设置）、`mahjong_helper.log`（日志）、`debug_output`（调试输出）会在 exe 所在目录自动生成。

### 方式二：从源码运行

需要 Python 3.8+（推荐 3.10+）。

```bash
git clone https://github.com/toki-2004/MahjongHelper.git
cd MahjongHelper
pip install -r requirements.txt
python main.py
```

> 源码版使用项目根目录下的 `templates/` 作为牌面模板；打包版内置在 `_internal/templates`。

---

## ⌨️ 快捷键

| 按键 | 功能 |
|------|------|
| `F2` | 进入框选模式（重新划定识别区域） |
| `F1` | 刷新识别（沿用当前区域，重新识别） |
| `ESC` | 取消当前操作（取消框选或关闭覆盖层） |

---

## 🧩 修正与自定义

- **框选技巧**：按 F2 后拖拽鼠标，尽量只框住一行完整牌面，减少背景和其他元素干扰，识别更稳定。
- **修正识别**：每张牌下方都有下拉栏（含全部 34 种牌）。选择正确牌后，程序会将该牌图像替换进对应模板并立即刷新识别。
- **自动模式**：右键托盘图标 → 设置，可切换为“自动 1s / 2s / 3s”，程序会定时识别当前区域。
- **模板目录**：源码版为 `templates/`，打包版为 `_internal/templates/`。在线修正会直接更新其中对应的牌面图片。

---

## ❓ 常见问题

- **热键没反应**：请以管理员身份运行；若被杀毒软件拦截，请将程序加入白名单。
- **识别为 0 张或出现 `?`**：框选区域太小或牌面显示过小，建议把图片/窗口缩放到 100% 以上再识别；`?` 表示该槽位未能识别，可用下拉栏手动修正。
- **识别数量不是 13~14 张**：框选时请尽量只包含一行牌面，避免把相邻牌、背景或其他区域一起框入。
- **切牌建议的原理**：程序为每张候选牌计算“打掉后剩余牌的组合权重”，权重越高说明剩余牌越整齐，因此推荐该牌。

---

## 📦 自行打包（开发者）

仓库自带 PyInstaller 配置（单目录模式、关闭 UPX，启动速度与源码运行相当）：

```bash
pip install pyinstaller
python -m PyInstaller --noconfirm --clean MahjongHelper.spec
```

产物位于 `dist/MahjongHelper/`，将**整个目录**压缩后即可分发。

---

## 📁 项目结构

```
main.py              主程序：界面、热键、识别线程
vision.py            图像识别：区域提取、分格对齐、模板匹配
template_manager.py  模板加载、矩阵预计算、在线修正
logic.py             切牌建议（组合权重动态规划）
templates/           34 张牌面模板（0.png ~ 33.png）
MahjongHelper.spec   PyInstaller 打包配置
```

---

## 📄 许可证

本项目采用 **MIT License**，详情见 [LICENSE](LICENSE)。

---

## 🙏 致谢

感谢开源社区的优秀库：PyQt5、OpenCV、NumPy、mss、keyboard 等。

---

**Enjoy your game!** 🀄️🎴
