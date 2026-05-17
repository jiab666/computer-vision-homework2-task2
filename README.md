# HW2 Task 2

这个目录保存了《计算机视觉 HW2》任务 2 的完整实验代码、训练结果与视频分析产物。当前最终实验主线已经确定为：

- 使用 `VisDrone2019-DET` 数据集
- 基于官方预训练权重 `YOLOv8n` 进行微调
- 在真实街景视频上结合 `ByteTrack` 完成多目标跟踪
- 基于 `track_id` 连续性实现越线计数
- 导出连续关键帧完成遮挡与 ID 稳定性分析

## 目录说明

- `src/convert_visdrone_to_yolo.py`
  将 VisDrone 原始标注转换为 YOLO 检测格式。

- `src/train_detector.py`
  使用 Ultralytics YOLO 训练/微调检测模型，并可选记录到 `SwanLab`。

- `src/track_and_count.py`
  对视频运行检测 + 多目标跟踪 + 越线计数。

- `src/export_key_frames.py`
  从跟踪结果视频中导出连续 3-4 帧，用于实验报告插图。

- `src/find_occlusion_segments.py`
  自动筛选更可能发生遮挡/密集交汇的候选片段。

- `src/render_test_sequence.py`
  可将 `test-dev` 连续帧序列导出成视频，主要用于流程验证。

- `src/visdrone_utils.py`
  公共工具函数与 VisDrone 类别映射。

- `REPORT_task2_final.md`
  任务 2 的实验报告。

## 环境配置

建议在独立环境中安装依赖。当前任务 2 实际使用的依赖安装方式如下：

```bash
conda create -n cvhw2_task2 python=3.10 -y
conda activate cvhw2_task2
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install swanlab
```

## 数据准备

当前脚本默认你的工作区中已经包含以下目录：

```text
archive/
├─ VisDrone2019-DET-train/
├─ VisDrone2019-DET-val/
├─ VisDrone2019-DET-test-dev/
└─ yolov9_finetuned.pt
```

虽然 `archive` 中保留了 `yolov9_finetuned.pt`，但本次正式实验主线不再使用它作为最终报告模型，而是采用官方 `YOLOv8n`。

首先将数据转为 YOLO 格式：

```bash
python src/convert_visdrone_to_yolo.py
```

转换后会生成：

```text
datasets/visdrone_yolo/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ visdrone_local.yaml
```

当前转换结果规模为：

- `train: 6471`
- `val: 548`
- `test: 1610`

## 检测模型训练

### 1. 正式训练命令

```bash
python src/train_detector.py ^
  --data datasets/visdrone_yolo/visdrone_local.yaml ^
  --weights yolov8n.pt ^
  --epochs 50 ^
  --batch 8 ^
  --imgsz 960 ^
  --workers 0 ^
  --device 0 ^
  --use-swanlab ^
  --swanlab-project cv-hw2-task2 ^
  --swanlab-experiment visdrone_yolov8n
```

说明：

- `YOLOv8n` 是本次正式作业主模型。
- `workers=0` 是在 Windows 环境下更稳的配置。
- 训练过程中使用 `SwanLab` 记录超参数和训练曲线。

### 2. 冒烟测试命令

如果只想先验证环境、显存和日志记录是否正常，可先运行轻量级测试：

```bash
python src/train_detector.py ^
  --data datasets/visdrone_yolo/visdrone_local.yaml ^
  --weights yolov8n.pt ^
  --epochs 3 ^
  --batch 4 ^
  --imgsz 640 ^
  --workers 0 ^
  --device 0 ^
  --use-swanlab ^
  --swanlab-project cv-hw2-task2 ^
  --swanlab-experiment smoke_test
```

### 3. 训练结果

正式训练结果保存在：

- `runs/task2_detect/visdrone_yolov8n/`

其中关键文件包括：

- [results.csv](runs/task2_detect/visdrone_yolov8n/results.csv)
- [results.png](runs/task2_detect/visdrone_yolov8n/results.png)
- [PR_curve.png](runs/task2_detect/visdrone_yolov8n/PR_curve.png)
- [P_curve.png](runs/task2_detect/visdrone_yolov8n/P_curve.png)
- [R_curve.png](runs/task2_detect/visdrone_yolov8n/R_curve.png)
- [F1_curve.png](runs/task2_detect/visdrone_yolov8n/F1_curve.png)
- [best.pt](runs/task2_detect/visdrone_yolov8n/weights/best.pt)

最终关键指标为：

- `Precision = 0.4680`
- `Recall = 0.3924`
- `mAP@0.5 = 0.3886`
- `mAP@0.5:0.95 = 0.2348`

## 测试视频与多目标跟踪

本次正式实验使用的真实测试视频为：

- `12208359_1920_1080_60fps.mp4`
- 来源：<https://www.pexels.com/video/pedestrians-crossing-the-street-27700659/>

视频信息：

- 分辨率：`1920x1080`
- 帧率：`59.94 fps`
- 总帧数：`1800`
- 时长：约 `30.03 秒`

### 正式跟踪与越线计数命令

```bash
python src/track_and_count.py ^
  --weights runs/task2_detect/visdrone_yolov8n/weights/best.pt ^
  --source 12208359_1920_1080_60fps.mp4 ^
  --tracker bytetrack.yaml ^
  --line 250,820,1670,820 ^
  --line-width 1 ^
  --font-size 0.35 ^
  --output-video artifacts/tracking/12208359_1920_1080_60fps_tracked.mp4 ^
  --output-csv artifacts/tracking/12208359_1920_1080_60fps_tracks.csv
```

输出结果：

- [12208359_1920_1080_60fps_tracks.csv](artifacts/tracking/12208359_1920_1080_60fps_tracks.csv)

最终越线计数结果：

- `Final crossing count: 38`

说明：

- 当前计数逻辑基于同一 `track_id` 的中心点跨越直线进行统计。
- 每个目标默认只计数一次，避免重复计数。

## 遮挡与 ID 稳定性分析

### 1. 导出关键帧

当前正式报告使用的是第 `600-603` 帧：

```bash
python src/export_key_frames.py ^
  --video artifacts/tracking/12208359_1920_1080_60fps_tracked.mp4 ^
  --start-frame 600 ^
  --num-frames 4 ^
  --output-dir artifacts/analysis/key_frames_600
```

导出结果为：

- [frame_00600.jpg](artifacts/analysis/key_frames_600/frame_00600.jpg)
- [frame_00601.jpg](artifacts/analysis/key_frames_600/frame_00601.jpg)
- [frame_00602.jpg](artifacts/analysis/key_frames_600/frame_00602.jpg)
- [frame_00603.jpg](artifacts/analysis/key_frames_600/frame_00603.jpg)

### 2. 分析结论

当前选取的 `600-603` 帧片段具有如下特点：

- 中部到右侧区域人流密集
- 多个行人目标彼此靠近并发生局部遮挡
- 多数目标在连续 4 帧中仍能保持较稳定的 `track_id`
- 远处小目标区域仍存在短时丢失和误匹配风险

这组关键帧已用于：

- [REPORT_task2_final.md](REPORT_task2_final.md)

## 结果文件位置

### 训练结果

- `runs/task2_detect/visdrone_yolov8n/`

### 跟踪与计数结果

- `artifacts/tracking/12208359_1920_1080_60fps_tracks.csv`

### 关键帧分析结果

- `artifacts/analysis/key_frames_600/`

### 报告与展示

- [REPORT_task2_final.md](REPORT_task2_final.md)

## 说明

本仓库当前内容已经与最终实验流程保持一致：

- 正式检测模型：`YOLOv8n`
- 正式测试视频：`12208359_1920_1080_60fps.mp4`
- 正式跟踪算法：`ByteTrack`
- 正式越线计数结果：`38`

如果后续更换测试视频，建议同步修改：

- 跟踪输出文件名
- 关键帧导出目录
- 实验报告中的视频与计数结果说明
