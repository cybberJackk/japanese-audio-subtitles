# Japanese Audio Subtitles

面向日语广播剧 / ボイスドラマ / ASMR 的全自动中日双语字幕生成工具。支持**本地命令行一键全自动闭环（Windows / Linux + Google Drive + Colab L4）**以及**浏览器 Colab Notebook 直接运行**。

## 工作流与核心设计

`本地音频目录` → `16kHz 单声道 FLAC 预处理（保留立体声声像平衡）` → `Google Drive 传输` → `Colab L4 四阶段双 ASR 引导精译` → `原音频同级目录落盘校验` → `自动清理云端临时文件并关闭 Colab 实例`

### 四阶段流水线（跨音轨按阶段批处理，每个模型每批仅加载一次）

1. **Stage 1 (`dual-asr`)**：按 450ms 句间停顿切分短窗，结合 `litagin/anime-whisper`（擅长声优语气、叹词与呼吸抗噪）和带声学提示词的 `openai/whisper-large-v3-turbo`（擅长数字、小数与片假名外来语）生成双路初稿。
2. **Stage 2 (`arbitrate-ja`)**：`Qwen3-14B` 先通读全轨双路初稿与作品背景，自动提取《全轨全局导引卡》（剧情脉络、角色人称与语气反差、场景动态同音词纠错、专有名词表），并在同专辑多音轨间共享，再 4 路并发校对输出黄金日文台本。
3. **Stage 3 (`align`)**：`Qwen3-ForcedAligner-0.6B` 词级声学强制对齐，无损投射原句标点与小数，并结合立体声声道方位（Stereo Pan）在多人对话切换时自动断句。
4. **Stage 4 (`translate-zh`)**：先由 `Qwen3-14B` 生成逐句 1:1 中文语义锚点（锁定敬语主客体方向与人称），再由 `ASMR-Qwen3.5-9B-Echo` 结合全局导引卡与动态术语表 4 路并发润色，保持原始声学时间轴不可篡改。

### 输出文件（直接保存在原音频同级目录）

- `<原音频文件名>.vtt`：简体中文 WebVTT 字幕（兼容主流本地与网页播放器自动挂载）
- `<原音频文件名>.ja.vtt`：日文对齐 WebVTT 字幕
- `<原音频文件名>.zh.txt` / `<原音频文件名>.ja.txt`：中/日纯文本台本

---

## 方式一：本地一键命令行（推荐）

本地电脑**零第三方 Python 依赖**（仅使用 Python 3.11+ 标准库），不会在音频文件夹内生成任何隐藏垃圾目录。

### 前置准备
1. 本地已安装 **Python 3.11+** 与 **`ffmpeg`**（已加入 `PATH`）。
2. 已配置 `colab` CLI（Windows 默认通过 WSL `Ubuntu-24.04` 调用）。
3. 首次使用执行一次 Google Drive OAuth 授权（凭据保存在本机 `%LOCALAPPDATA%\JapaneseAudioSubtitles\drive-token.json`）：
   ```powershell
   .\colab_start.cmd --auth-client-secret "C:\path\to\client_secret.json"
   ```
   *(首次在空 Google Drive 账号上运行时，云端会自动从 HuggingFace 下载模型并持久缓存到 `MyDrive/japanese-subtitles/models/`，之后每次启动仅需约 1~2 分钟从云盘恢复。)*

### 常用命令

```powershell
# 处理整个音频文件夹（递归扫描，自动跳过已有 .vtt 的音轨）
.\colab_start.cmd "D:\Audio\DramaAlbum01"

# 一次排队多个文件夹或单条音频（共享同一次 L4 模型加载）
.\colab_start.cmd "D:\Audio\AlbumA" "D:\Audio\AlbumB"

# 指定自定义术语表或背景设定文件
.\colab_start.cmd "D:\Audio\MyAudio" --metadata "D:\Audio\MyAudio\glossary.json"

# 预览待处理音轨列表，不上传、不启动 GPU
.\colab_start.cmd "D:\Audio\DramaAlbum01" --dry-run

# 强制覆盖重跑本地已有字幕（加 --fresh 可同时忽略云端未完成断点）
.\colab_start.cmd "D:\Audio\DramaAlbum01\01.wav" --overwrite --fresh
```

---

## 针对特定领域音声的优化指南（自定义词表与背景设定）

通用开源大模型与语音识别在面对**特定垂直领域音声**（如科幻机甲、医疗健康、奇幻设定、历史剧或特定题材同人作品）时，往往会因为缺乏领域专有词汇或人物背景，把罕见术语误转写为日常同音词，或在多义词上出现字面误译。

你可以通过以下**仅保存在本地（已被 `.gitignore` 忽略，不会上传到 Git 仓库）**的方式为特定领域注入知识：

### 1. 全局本地基础词表（推荐用于特定领域的通用高频词）
在项目根目录创建 `glossary.local.json`（或放在 `%LOCALAPPDATA%\JapaneseAudioSubtitles\default_glossary.json`）。调度器在每次运行时会自动加载并合并该词表，同时将其中的日文词汇注入 Whisper 的声学引导提示词中：

```json
[
  {"ja": "当機", "zh": "本机", "note": "科幻仿生人自称（防止ASR误识为陶器/当期）"},
  {"ja": "マスター", "zh": "主人", "note": "通用称呼"}
]
```

### 2. 单部作品专属设定与词表（`metadata.json` 或 `glossary.json`）
在具体音频文件夹（或其父目录）下放置 `glossary.json`：
```json
{
  "アリス": "爱丽丝",
  "엘릭サー": "灵药"
}
```
或者放置包含剧情背景简介的 `metadata.json`（向 Stage 2 提供作品世界观与人物关系，能显著提升模型自动推断同音错字和主客体人称的准确率）：
```json
{
  "work_title": "作品标题",
  "synopsis_ja": "日文背景简介（介绍说话人身份、场景设定与角色关系，帮助模型根据语境自动消歧同音词）",
  "base_glossary": [
    {"ja": "アリス", "zh": "爱丽丝", "note": "角色名"}
  ]
}
```

### 3. 本地自动元数据钩子（`metadata_hook.local.py`）
如果你希望根据本地文件夹命名规则自动从外部数据库或本地索引拉取作品标题与日文简介，可在项目根目录创建 `metadata_hook.local.py`（同样被 `.gitignore` 忽略），实现 `lookup_work_metadata(audio_path, app_data_dir, cache_mem) -> dict` 即可自动注入 `work_title` 与 `synopsis_ja`。

---

## 方式二：浏览器 Colab Notebook 直接运行

如果不使用本地命令行，可直接在 Google Colab 打开 [`colab_subtitles.ipynb`](./colab_subtitles.ipynb)，选择 **L4 GPU** 按顺序运行 3 个单元格即可。

---

## 项目结构

```text
japanese-audio-subtitles/
├── colab_start.cmd              # Windows 快捷入口（支持直接拖拽文件夹到脚本图标上运行）
├── colab_subtitles.py           # 本地无依赖调度器（扫描、转码、Drive 传输、Colab 守护、校验清理）
├── colab_subtitles.ipynb        # 浏览器端 Google Colab 一键 Notebook
├── runtime/                     # 云端推理与存储核心模块
│   ├── bootstrap.py             # 模型与 llama.cpp 的 Google Drive 持久缓存与快速恢复
│   ├── drive_store.py           # 零依赖 Google Drive v3 HTTP API / FUSE 存储层
│   ├── pipeline.py              # 四阶段双 ASR + Qwen3-14B 仲裁 + ForcedAligner + 9B 精译流水线
│   ├── subtitle_utils.py        # WebVTT 读写校验、标点无损投射、防复读压缩与 JSON 解析
│   └── worker.py                # Colab 批量任务进程管理与阶段断点同步
├── tests/
│   └── test_subtitle_utils.py   # 核心字幕处理与断点逻辑单元测试
└── docs/
    └── architecture-and-benchmarks.md  # L4 硬件基准测试与架构说明
```
