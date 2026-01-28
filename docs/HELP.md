# 📖 **Configuration Guide**

<p align="center">
  <a href="HELP.md">🇺🇸 <b>English</b></a> | <a href="HELP_ru.md">🇷🇺 Русский</a>
</p>

> Complete explanation of all YDS configuration files, their structure, and usage.

---

## 📑 **Quick Navigation**

| Section | Topic |
|---------|-------|
| [⚙️ File Location](#-configuration-files-location) | Where config files are |
| [🔧 Config #1](#-config-1-configsconfig.json--main-configuration) | Main settings (GUI, data, training) |
| [🎬 Config #2](#-config-2-configsconfigstreamcutjson--streamcut) | StreamCut VOD processing |
| [🎯 Configuration Examples](#-step-by-step-configuration-examples) | Real-world scenarios |
| [🔧 Troubleshooting](#-troubleshooting-checklist) | Fix common issues |
| [📊 Performance](#-performance-optimization) | Speed & optimization tips |
| [🎨 YOLO Models](#-working-with-different-yolo-models) | Model selection & usage |
| [💾 Export & Deploy](#-model-export--deployment) | Save & use trained models |

---

## 📂 **Configuration Files Location**

```
YolovTrainGui/
├── configs/
│   ├── config.json           # ⚙️ Main config (GUI, data collection, training)
│   └── configStreamCut.json  # 🎬 StreamCut config (VOD processing)
```

**💡 Good to know:** Most parameters can be changed in the GUI, but configuration files provide:
- ✅ Precise control over all settings
- ✅ Save presets between sessions
- ✅ Run scripts directly from CLI

---

## � **Config #1: `configs/config.json`** — Main Configuration

**Used for:**
- 📸 **Data Collection** — Semi-automatic screen capture settings
- 🖌️ **Label Tool** — Label verification and correction
- 📂 **Dataset Splitting** — Train/val/test partitioning
- ⚙️ **Training** — Default hyperparameters
- 🎨 **UI** — Interface appearance

<details>
<summary><b>📋 Click to expand JSON example</b></summary>

### **Essential Parameters**

```json
{
  "model_path": "models/sunxds_0.7.6.pt",
  "classes": ["player", "head", "weapon"],
  
  "grabber": {
    "crop_size": 0.8,
    "width": 640,
    "height": 640
  },
  
  "output_folder": "dataset_output",
  "save_interval": 3,
  "detection_threshold": 0.5,
  
  "train_defaults": {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.01,
    "patience": 15,
    "mosaic": 1.0,
    "mixup": 0.1,
    "amp": true
  },
  
  "ui": {
    "font_size": 10,
    "theme": "light_cyan"
  }
}
```

</details>

### **Parameter Explanation**

#### **📸 Data Collection Settings**
- **model_path** — YOLO model for detection (affects data collection only, not training)
- **classes** — Objects to capture (order = class ID in dataset)
- **grabber.crop_size** — Screen area to capture (0.8 = 80% of center)
- **grabber.width/height** — Fixed capture resolution
- **detection_threshold** — Min confidence to save frame (0-1)
- **save_interval** — Seconds between saves (prevents duplicates)

#### **📂 Output & Storage**
- **output_folder** — Where to save captured images/labels
- **data_folder** — Folder for label tool and splitting

#### **⚙️ Training Defaults**
These values appear as defaults in the Training GUI:

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| **epochs** | 50 | 10-500 | Training iterations |
| **imgsz** | 640 | 320-1280 | Input image size |
| **batch** | 16 | 1-128 | Samples per step |
| **optimizer** | SGD | SGD/Adam/AdamW | Gradient descent type |
| **lr0** | 0.01 | 0.0001-0.1 | Learning rate |
| **patience** | 15 | 5-50 | Early stopping |
| **mosaic** | 1.0 | 0-1 | Multi-scale training |
| **mixup** | 0.1 | 0-1 | Image mixing probability |
| **amp** | true | - | Automatic Mixed Precision |

#### **🎨 UI Settings**
- **font_size** — GUI font size (8-20)
- **theme** — Color scheme (light_cyan, dark_blue, etc.)

---

## 🎬 **Config #2: `configs/configStreamCut.json`** — StreamCut

Manages the VOD processing pipeline: **Download → Split → Infer → Save**

<details>
<summary><b>📋 Click to expand JSON example</b></summary>

### **Essential Parameters**

```json
{
  "video_sources": [
    "https://www.twitch.tv/videos/2522936875",
    "https://www.twitch.tv/videos/2524662899"
  ],
  "use_selected_only": true,
  
  "raw_stream_folder": "stream/raw_streams",
  "chunks_folder": "stream/chunks",
  "output_folder": "stream/dataset",
  
  "model_path": "models/sunxds_0.7.6.pt",
  "classes": ["player", "head"],
  "detection_threshold": 0.7,
  "time_interval": 3,
  
  "max_download_workers": 3,
  "split_workers": 4,
  "process_workers": 8,
  "save_workers": 2,
  
  "ffmpeg_path": "utils/ffmpeg/bin/ffmpeg.exe",
  "download_archive": "stream/downloaded.txt",
  "pause_after_download": true
}
```

</details>

### **Parameter Explanation**

#### **🎥 Video Sources**
- **video_sources** — List of all Twitch VOD URLs to process
- **use_selected_only** — If true, only process selected URLs (set via GUI)

#### **📁 Folders & Paths**
- **raw_stream_folder** — Where downloaded VODs are stored
- **chunks_folder** — Temporary folder for video segments
- **output_folder** — Final dataset location (images/ + labels/)
- **ffmpeg_path** — Path to ffmpeg executable

#### **🤖 Detection & Model**
- **model_path** — YOLO model for detection
- **classes** — Which objects to save (others ignored)
- **detection_threshold** — Min confidence to save frame (0-1)
- **time_interval** — Check every N frames (3 = ~10 frames/sec at 30fps)

#### **⚡ Worker Threads** (Critical for Performance!)

| Worker | Purpose | Recommendation |
|--------|---------|-----------------|
| **max_download_workers** | How many videos download simultaneously | 2-3 (⚠️ max 3 to avoid Twitch ban) |
| **split_workers** | FFmpeg threads for video segmentation | 🔧 CPU core count |
| **process_workers** | GPU threads running YOLO inference | 🎮 1-4 (GPU dependent) |
| **save_workers** | Disk I/O threads for saving frames/labels | 💾 1-2 |

**🚀 Tuning Guide:**
- More download workers = faster (but risk getting banned if >3)
- More split workers = use CPU cores available
- More process workers = better GPU utilization (but uses more VRAM)
- More save workers = rarely needed (I/O usually not bottleneck)

#### **⚙️ Other Settings**
- **download_archive** — File tracking already-downloaded VODs (avoids re-downloading)
- **pause_after_download** — Ask before processing after download completes
- **time_interval** — Detection frequency:
  - `1` = Every frame (comprehensive but slow)
  - `3-5` = **Optimal** (6-10 checks/sec at 30fps) ⭐
  - `30` = Once per second

---

## ⚠️ **Common Configuration Issues**

| Problem | Cause | Solution |
|---------|-------|----------|
| No objects detected | model_path wrong or threshold too high | Check path, lower threshold to 0.3-0.4 |
| Out of memory | batch size too large or process_workers too high | Reduce batch or process_workers |
| Slow data collection | interval too short or model too large | Increase save_interval to 5+ seconds |
| Slow StreamCut | Not enough workers | Increase split_workers and process_workers (if hardware allows) |
| Twitch ban | Too many download workers | Keep max_download_workers ≤ 3 |
| Labels missing classes | class_map incorrect | Verify model class indices match your mapping |

---

## 💡 **Recommended Presets**

### **Fast Data Collection (Speed)**
```json
{
  "save_interval": 1,
  "detection_threshold": 0.3,
  "crop_size": 0.9,
  "batch": 32
}
```

### **High Quality (Accuracy)**
```json
{
  "save_interval": 5,
  "detection_threshold": 0.7,
  "crop_size": 0.6,
  "epochs": 100,
  "patience": 25
}
```

### **StreamCut on Weak GPU (GTX 1060)**
```json
{
  "process_workers": 1,
  "split_workers": 2,
  "max_download_workers": 2,
  "time_interval": 30,
  "batch": 8
}
```

### **StreamCut on Strong GPU (RTX 3080+)**
```json
{
  "process_workers": 8,
  "split_workers": 8,
  "max_download_workers": 3,
  "time_interval": 3,
  "batch": 32
}
```

---

## 🔗 **Config File Relationships**

```
config.json
├─ Data Collection → Uses model_path for detection
├─ Label Tool → Uses output_folder + classes
├─ Dataset Split → Reads from data_folder, writes to split location
├─ Training Tab → Uses train_defaults as starting point
└─ UI → Uses ui.font_size and ui.theme

configStreamCut.json
├─ StreamCut Tool → Uses all parameters
└─ Advanced Settings → Tuned by user via config
```

---

## 🎯 **Step-by-Step Configuration Examples**

### **Scenario 1: Quick Gaming Content Creator Setup**
**Goal:** Build detector for Valorant rank detection in 2 hours

1. **config.json:**
   ```json
   {
     "model_path": "models/yolov8s.pt",
     "classes": ["rank_banner", "agent"],
     "detection_threshold": 0.4,
     "save_interval": 1,
     "grabber": {"width": 640, "height": 640, "crop_size": 0.7},
     "train_defaults": {
       "epochs": 30,
       "batch": 32,
       "imgsz": 640
     }
   }
   ```

2. **Process:**
   - Use screen capture for 30 minutes (will get ~1500 images)
   - Verify 50 random samples in label tool (5 min)
   - Train for 30 epochs (30 min on GPU)
   - Export and test

### **Scenario 2: High-Quality Multi-Game StreamCut Mining**
**Goal:** Build dataset from 10 Twitch VODs with maximum quality

1. **configStreamCut.json:**
   ```json
   {
     "detection_threshold": 0.75,
     "time_interval": 5,
     "max_download_workers": 2,
     "split_workers": 4,
     "process_workers": 4,
     "pause_after_download": true
   }
   ```

2. **Process:**
   - Add 10 VOD links to video_sources
   - Start processing (will take ~2-3 hours depending on video length)
   - Review generated labels in label tool
   - Adjust classes if needed

### **Scenario 3: Research/ML Experimentation**
**Goal:** Iterate quickly on labeled dataset

1. **config.json focus on:**
   ```json
   {
     "save_interval": 3,
     "output_folder": "dataset_output/experiment_1",
     "train_defaults": {
       "epochs": 50,
       "imgsz": 416,
       "batch": 8
     }
   }
   ```

2. **Workflow:**
   - Capture base dataset
   - Train v1
   - Analyze errors
   - Capture additional hard examples
   - Train v2 with more data
   - Compare metrics

---

## 🔧 **Troubleshooting Checklist**

### **Data Collection Issues**

**Q: No objects detected during screen capture**
- [ ] Is model_path file actually present? (`models/` folder)
- [ ] Is detection_threshold too high? (try 0.3)
- [ ] Are you trying to detect correct objects for the model?
- [ ] Is screen resolution at least 640x480?
- **Fix:** Reduce threshold → 0.3, verify model exists, test with known objects

**Q: Capture is very slow**
- [ ] Is save_interval too short? (try 3-5)
- [ ] Is model too large? (use yolov8n or yolov10n instead)
- [ ] Is crop_size too large? (0.8 or less)
- [ ] Is GPU being used? (check if CUDA is installed)
- **Fix:** Increase save_interval, use smaller model, check CUDA installation

**Q: Captured labels are incorrect**
- [ ] Does class_map match your model's class order?
- [ ] Is detection_threshold catching false positives? (increase to 0.6+)
- [ ] Are bounding boxes in wrong format? (should be normalized 0-1)
- **Fix:** Verify config classes, use label tool to manually fix, train new model

### **Training Issues**

**Q: Training crashes with CUDA out of memory**
- Reduce batch size: `"batch": 8` (from 16)
- Reduce image size: `"imgsz": 416` (from 640)
- Clear old training runs: Delete `runs/` folder

**Q: Model accuracy is low**
- Not enough data? Collect 2000+ images minimum
- Too much data imbalance? Make sure classes are well-represented
- Bad labels? Review 100+ samples in label tool
- Learning rate wrong? Try `"lr0": 0.001` for fine-tuning

**Q: Training is very slow**
- Use smaller model: `yolov8n` instead of `yolov8m`
- Reduce image size: `"imgsz": 416`
- Increase batch size if VRAM allows
- Check if GPU is being used: Look at GPU utilization in task manager

### **StreamCut Issues**

**Q: Videos not downloading**
- [ ] Are URLs correct Twitch VOD links?
- [ ] Do you have internet connection?
- [ ] Is yt-dlp updated? (`pip install -U yt-dlp`)
- [ ] Are download workers > 3? (reduce to avoid ban)
- **Fix:** Update yt-dlp, check URLs, verify internet

**Q: Getting Twitch rate limited/banned**
- [ ] Is max_download_workers > 3?
- [ ] Are you downloading too many videos too quickly?
- **Fix:** Set `max_download_workers: 2`, wait before starting new batch

**Q: StreamCut processing is very slow**
- Increase split_workers to match CPU cores
- Increase process_workers (if GPU available)
- Increase time_interval to 5-10 (checks less frequently)
- Use weaker model for detection (yolov8n instead of yolov12x)

**Q: No labels generated from StreamCut**
- [ ] Is model_path correct?
- [ ] Are classes in configStreamCut.json correct?
- [ ] Is detection_threshold too high? (try 0.3)
- [ ] Are time_interval frames actually containing your objects?
- **Fix:** Check threshold, verify model, test detection on sample frames

---

## 📊 **Performance Optimization**

### **For Data Collection (Screen Capture)**

| Bottleneck | Solution | Impact |
|-----------|----------|--------|
| Slow detection | Use smaller model (yolov8n) | ~3x faster, slightly less accurate |
| Slow I/O | Increase save_interval | ~2x faster collection, fewer frames |
| High RAM usage | Reduce crop_size (0.6 instead of 0.8) | Lower resolution, faster but less detail |

### **For Training**

| Bottleneck | Solution | Impact |
|-----------|----------|--------|
| Out of memory | Reduce batch size 32→16→8 | Slower training, but works |
| Slow training | Reduce imgsz 640→416 | ~4x faster, slight accuracy loss |
| Slow GPU | Use smaller model (n/s instead of m/l) | Much faster, less accurate |

### **For StreamCut**

| Bottleneck | Solution | Impact |
|-----------|----------|--------|
| Slow download | max_download_workers: 3 (max!) | ~3x faster, risk of ban |
| Slow split | split_workers: [CPU cores] | Linear speedup |
| Slow inference | time_interval: 10-30 | ~2-5x faster, fewer frames |
| Slow save | Usually not bottleneck | - |

---

## 🎨 **Working with Different YOLO Models**

### **Model Selection Guide**

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| yolov8n | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | Real-time, weak hardware |
| yolov8s | ⚡⚡ | ⭐⭐⭐⭐ | 4GB | Balanced, most use |
| yolov8m | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | High accuracy needed |
| yolov12x | 🐌 | ⭐⭐⭐⭐⭐ | 24GB | Research, highest quality |

### **Changing Models in Config**

```json
{
  "model_path": "models/yolov8n.pt",  // Change this
  "classes": ["player", "weapon"],     // Make sure classes match your data!
  "detection_threshold": 0.5
}
```

**Important:** Each model has different:
- Class indices (what class 0, 1, 2 mean)
- Detection confidence calibration
- Speed/accuracy tradeoff

---

## 💾 **Model Export & Deployment**

### **After Training**

Best model saved to: `runs/detect/train/weights/best.pt`

### **Export to ONNX** (for benchmarking)
```
From Tools menu → Export Model → Select run folder → Choose ONNX format
```
Creates `.onnx` file for cross-platform inference

### **Using Trained Model**

1. Copy `best.pt` to `models/` folder
2. Update config.json: `"model_path": "models/your_model.pt"`
3. Use in:
   - Screen capture for data collection
   - StreamCut for VOD mining
   - Label tool for verification

---

1. **Always backup config.json before major changes**
2. **Use GUI to discover available options** (configs only store what's set)
3. **Twitch cookies format:** Must be Netscape format (export from browser extension)
4. **model_path in config.json doesn't affect training** — Training uses model selected in GUI
5. **time_interval in StreamCut:** If unsure, use 3-5 for good balance

---

## 🆘 **Getting Help**

- Check README.md for workflow overview
- See README_ru.md for Russian documentation  
- Review common issues above before reporting bugs

---

<div align="center">

**Configuration files are the advanced settings. Most users control everything via GUI!**

</div>
