<div align="center">

# ⚙️ **YDS — Configuration Reference**

Concise guide to all `config.json` and `configStreamCut.json` parameters.

<p align="center">
  <a href="HELP.md">🇺🇸 <b>English</b></a> | <a href="HELP_ru.md">🇷🇺 Русский</a>
</p>

</div>

## 📌 **Quick Navigation**

- [Where files are](#where-configuration-files-are-located)
- [Output Structure](#output-structure-yolo-format)
- [Quick Start](#quick-start--choose-your-task)
- [config.json Reference](#configjson-reference)
- [configStreamCut.json Reference](#configstreamcutjson-reference)
- [Important Relationships](#important-relationships--gotchas)
- [Presets](#presets--ready-to-use-configs)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

---

## 📂 **Where Configuration Files Are Located**

```
YolovTrainGui/
├── configs/
│   ├── config.json              # Main config (screen capture, training, UI)
│   └── configStreamCut.json     # StreamCut config (Twitch VOD, workers)
```

| File | Purpose | Modified by |
|------|---------|-------------|
| **config.json** | Screen capture, training defaults, UI | GUI or JSON editor |
| **configStreamCut.json** | Twitch VOD download, processing, parallel workers | GUI or JSON editor |

---

## 📦 **Output Structure (YOLO format)**

### **Screen Capture Output**
```
output_folder/ (default: "dataset_output")
├── images/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
└── labels/
    ├── frame_001.txt
    ├── frame_002.txt
    └── ...
```

### **StreamCut Output**
```
stream/dataset/ (or configStreamCut.json output_folder)
├── images/
│   ├── chunk_0_frame_100.png
│   └── ...
└── labels/
    ├── chunk_0_frame_100.txt
    └── ...
```

### **Label Format (YOLO standard)**
Each `.txt` file contains one line per detection:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id` — Index in your `classes` array (0, 1, 2, ...)
- `x_center, y_center, width, height` — Normalized to 0–1 (independent of image size)

**Example:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
```
Means: class 0 at center, class 1 at left-bottom.

---

## 🚀 **Quick Start — Choose Your Task**

### **"I want to collect screen data"**
Edit `config.json`:
- `model_path` → which YOLO model to use
- `detection_threshold` → confidence cutoff (try 0.3–0.5)
- `save_interval` → minimum seconds between saves
- `grabber.crop_size` → screen crop (0.6 = 60%)

**Go to:** [Data Collection Parameters](#-data-collection)

### **"I want to download & process Twitch VODs"**
Edit `configStreamCut.json`:
- `video_sources` → add Twitch URLs
- `model_path` → which YOLO model to use
- `time_interval` → inference every N frames (5 = every 5th frame)
- `process_workers` → parallel inference (4 = 4 threads)

**Go to:** [StreamCut Parameters](#-download--processing)

### **"My detection is wrong"**
Check:
- `classes` → are class names correct?
- `class_map` → does it match your model?
- `detection_threshold` → too high? try 0.2–0.3

**Go to:** [Important Relationships](#-important-relationships--gotchas)

### **"My GPU is slow"**
Try:
- Screen capture: use `yolov8n.pt` instead of larger model
- StreamCut: reduce `process_workers` or increase `time_interval`

**Go to:** [Performance Tuning](#-performance-tuning)

---

## ⚙️ **config.json Reference**

### **📸 Data Collection**

#### `model_path`
- **Type:** string (file path)
- **Default:** `"models/yolov8n.pt"`
- **Used in:** Screen Capture
- **Meaning:** Which YOLO model to use for detecting objects
- **Tip:** Smaller models = faster but less accurate. Use `yolov8n` for speed, `yolov8m` for accuracy.

#### `detection_threshold`
- **Type:** float (0.0–1.0)
- **Default:** `0.5`
- **Used in:** Screen Capture (affects what gets saved)
- **Meaning:** Minimum confidence to save a detection
- **Tip:** If nothing saves, try `0.3–0.4`. If too many false positives, increase to `0.6–0.7`.

#### `save_interval`
- **Type:** integer (seconds)
- **Default:** `2`
- **Used in:** Screen Capture
- **Meaning:** Minimum delay between saves (prevents duplicate frames)
- **Tip:** Increase to `5–10` if dataset is too large. Decrease to `1` for dense sampling.

---

### **🖼️ Screen Capture (grabber)**

#### `grabber.crop_size`
- **Type:** float (0.0–1.0)
- **Default:** `0.8`
- **Used in:** Screen Capture
- **Meaning:** Center crop percentage (0.8 = crop to 80% of minimum screen dimension)
- **Tip:** Smaller values exclude screen edges. Use `0.6–0.8` for gameplay.

#### `grabber.width` / `grabber.height`
- **Type:** integer (pixels)
- **Default:** `1920` / `1080`
- **Used in:** Screen Capture
- **Meaning:** Capture resolution
- **Tip:** Higher = more detail but slower. Use 1280×720 for speed, 1920×1080 for detail.

---

### **📁 Folders & Paths**

#### `output_folder`
- **Type:** string (path)
- **Default:** `"dataset_output"`
- **Used in:** Screen Capture
- **Meaning:** Where to save captured `images/` and `labels/`

#### `data_folder`
- **Type:** string (path)
- **Default:** `"datasets"`
- **Used in:** Split Dataset, Label Verification
- **Meaning:** Default dataset folder for all tools

#### `label_data_folder`
- **Type:** string (path)
- **Default:** `""` (falls back to `data_folder`)
- **Used in:** Label Verification tool
- **Meaning:** Which folder to open in the label tool

#### `last_data_yaml`
- **Type:** string (path)
- **Default:** `""`
- **Used in:** Training GUI
- **Meaning:** Last selected `data.yaml` (auto-filled, read-only)

---

### **🏷️ Classes**

#### `classes`
- **Type:** array of strings
- **Default:** `["class1", "class2"]`
- **Used in:** Screen Capture, Label Verification, Training
- **Meaning:** List of object class names. **Order matters!** Index in labels = position in this array.
- **Tip:** If class order doesn't match model output, use `class_map` to remap.

#### `class_map`
- **Type:** object (key-value pairs)
- **Default:** `{}`
- **Used in:** Screen Capture
- **Meaning:** Maps model class indices to dataset classes
- **Example:** `{"0": "player", "7": "head"}` = model class 0 → save as "player", model class 7 → save as "head"
- **Tip:** Generate via GUI "Show Classes from Model" button.

---

### **⚙️ Training Defaults**

#### `train_defaults`
- **Type:** object
- **Default:** See below
- **Used in:** Training GUI (fills in default values)
- **Meaning:** Pre-filled training parameters (all overridable in GUI)

Key parameters:
- `epochs` (int) — training iterations
- `imgsz` (int) — input image size (320, 416, 640)
- `batch` (int) — batch size (reduce if GPU memory limited)
- `patience` (int) — early stopping (stop if no improvement for N epochs)
- `amp` (bool) — Automatic Mixed Precision (faster on modern GPUs)
- `mosaic`, `mixup`, `fliplr`, `hsv_h`, etc. — augmentation parameters

---

### **🎨 UI Settings**

#### `ui.font_size`
- **Type:** integer (8–14)
- **Default:** `10`
- **Used in:** GUI appearance
- **Meaning:** Font size for all GUI windows

#### `ui.theme`
- **Type:** string
- **Default:** `"light_blue"`
- **Used in:** GUI appearance
- **Meaning:** Qt-material theme name (light_blue, dark_teal, light_cyan, etc.)

#### `ui.show_capture_window`
- **Type:** boolean
- **Default:** `true`
- **Used in:** Screen Capture
- **Meaning:** Show OpenCV preview window during capture
- **Tip:** Set to `false` if preview causes performance issues.

---

## 🎬 **configStreamCut.json Reference**

### **📥 Download & Sources**

#### `video_sources`
- **Type:** array of strings (URLs)
- **Default:** `[]`
- **Used in:** StreamCut GUI list
- **Meaning:** All Twitch VOD URLs to potentially download
- **Example:** `["https://www.twitch.tv/videos/1234567890"]`

#### `selected_video_sources`
- **Type:** array of integers (indices)
- **Default:** `[]`
- **Used in:** StreamCut
- **Meaning:** Which URLs from `video_sources` are selected (only used if `use_selected_only: true`)
- **Example:** `[0, 2]` = process 1st and 3rd URL

#### `use_selected_only`
- **Type:** boolean
- **Default:** `false`
- **Used in:** StreamCut
- **Meaning:** If true, process only `selected_video_sources`. If false, process all `video_sources`.

#### `download_quality`
- **Type:** string
- **Default:** `"720p"`
- **Used in:** StreamCut download
- **Meaning:** Maximum quality to download (passed to yt-dlp)
- **Options:** `"best"`, `"720p"`, `"480p"`, `"360p"`

#### `twitch_cookies_path`
- **Type:** string (path)
- **Default:** `"Cookies/cookies.txt"`
- **Used in:** StreamCut download
- **Meaning:** Netscape cookies file (required for private or age-gated VODs)

#### `raw_stream_folder`
- **Type:** string (path)
- **Default:** `"stream/raw_streams"`
- **Used in:** StreamCut
- **Meaning:** Where VOD files are downloaded

#### `download_archive`
- **Type:** string (path)
- **Default:** `"stream/.download_archive"`
- **Used in:** StreamCut
- **Meaning:** File that tracks downloaded URLs. Used by **Sync** to skip already-downloaded VODs.

---

### **🔄 Processing Pipeline**

#### `chunks_folder`
- **Type:** string (path)
- **Default:** `"stream/chunks"`
- **Used in:** StreamCut
- **Meaning:** Temporary folder for `.ts` segments during processing

#### `chunks_per_stream`
- **Type:** integer
- **Default:** `10`
- **Used in:** StreamCut
- **Meaning:** How many segments to split each VOD into
- **Tip:** More segments = more parallel processing but more ffmpeg overhead.

#### `ffmpeg_path`
- **Type:** string (path)
- **Default:** `"utils/ffmpeg/bin/ffmpeg.exe"`
- **Used in:** StreamCut
- **Meaning:** Path to ffmpeg executable (used for splitting VODs)

#### `output_folder`
- **Type:** string (path)
- **Default:** `"stream/dataset"`
- **Used in:** StreamCut
- **Meaning:** Final dataset output (`images/` and `labels/`)

#### `time_interval`
- **Type:** integer (frames)
- **Default:** `5`
- **Used in:** StreamCut
- **Meaning:** Run inference every N frames (5 = check every 5th frame)
- **Tip:** Higher values = faster processing but fewer frames. Lower = slower but more dense sampling.

#### `resume_info_file`
- **Type:** string (path)
- **Default:** `"stream/.resume_info"`
- **Used in:** StreamCut
- **Meaning:** File that tracks which chunks have been processed. Used to resume interrupted runs.
- **Tip:** Delete this file to reprocess all chunks.

---

### **🤖 Detection & Model**

#### `model_path`
- **Type:** string (path)
- **Default:** `"models/yolov8n.pt"`
- **Used in:** StreamCut inference
- **Meaning:** YOLO model used for detecting objects in VOD frames
- **Tip:** Must exist before running StreamCut.

#### `detection_threshold`
- **Type:** float (0.0–1.0)
- **Default:** `0.3`
- **Used in:** StreamCut
- **Meaning:** Minimum confidence to save a detection

#### `classes`
- **Type:** array of strings
- **Default:** `[]`
- **Used in:** StreamCut
- **Meaning:** Only these classes are kept in the output dataset
- **Tip:** If empty, all detected classes are saved.

---

### **⚡ Parallel Workers & Performance**

#### `max_download_workers`
- **Type:** integer (1–8)
- **Default:** `2`
- **Used in:** StreamCut download
- **Meaning:** Parallel Twitch downloads
- **⚠️ Important:** Keep ≤ 2–3 to avoid Twitch rate limiting!

#### `split_workers`
- **Type:** integer (1–8)
- **Default:** `4`
- **Used in:** StreamCut (ffmpeg)
- **Meaning:** Parallel VOD segment splitting

#### `process_workers`
- **Type:** integer (1–8)
- **Default:** `4`
- **Used in:** StreamCut inference
- **Meaning:** Parallel YOLO inference threads
- **Tip:** Increase if GPU underutilized. Decrease if CPU maxes out.

#### `save_workers`
- **Type:** integer (1–8)
- **Default:** `4`
- **Used in:** StreamCut
- **Meaning:** Parallel disk writes (saving images/labels)

#### `pause_after_download`
- **Type:** boolean
- **Default:** `true`
- **Used in:** StreamCut GUI
- **Meaning:** Ask for confirmation before processing after download completes

---

## ⚠️ **Important Relationships & Gotchas**

### **1. `model_path` in config.json vs Training**
❌ **Common mistake:** Editing `model_path` in `config.json` doesn't affect Training tab.

✅ **Why:** Training uses the model selected in the **Training GUI** + `data.yaml`. `config.json` model_path is **only for screen capture detection**.

### **2. `classes` Order = Class ID**
❌ **Common mistake:** Changing order of classes in the middle of dataset collection.

✅ **Why:** The position in the `classes` array becomes the label ID. If you reorder, old labels won't match.

**Solution:** Define classes upfront and don't reorder.

### **3. `class_map` & Model Indices**
**Use `class_map` when:** Your model outputs class indices (0, 7, 15...) but you want to map them to your dataset class names.

**How it works:**
1. Model outputs class index (e.g., 7)
2. `class_map` converts it to a class name (e.g., `"7": "head"`)
3. YDS finds "head" in your `classes` array and gets its position (index)
4. That position becomes the `class_id` in the saved label

**Example:**
```json
"classes": ["player", "head", "weapon"],
"class_map": {"0": "player", "7": "head"}
```

Result:
- Model output 0 → maps to "player" → finds at `classes[0]` → saves as class_id `0`
- Model output 7 → maps to "head" → finds at `classes[1]` → saves as class_id `1`
- Model output 15 → not in class_map → ignored

### **4. `download_archive` & StreamCut Sync**
**How it works:** StreamCut tracks downloaded URLs in `download_archive`. Clicking **Sync** skips already-downloaded VODs.

**If Sync doesn't work:**
1. Check that `download_archive` file exists
2. Check that `raw_stream_folder` contains the VOD files
3. Verify URLs are identical

### **5. `time_interval` is in FRAMES, not seconds**
❌ **Common mistake:** Setting `time_interval: 30` expecting 30 seconds.

✅ **Reality:** `time_interval: 30` = check every 30th frame.

### **Avoid Twitch Rate Limits**
If you get "rate limit" errors:

```json
{
  "max_download_workers": 1,
  "pause_after_download": true
}
```

---

## 🔧 **Troubleshooting**

| Problem | Cause | Solution |
|---------|-------|----------|
| Nothing saves in screen capture | `detection_threshold` too high | Lower to 0.3–0.4 |
| Too many false positives | `detection_threshold` too low | Raise to 0.6–0.8 |
| Label tool opens wrong folder | `label_data_folder` not set | Set in config.json |
| StreamCut doesn't skip downloaded VODs | `download_archive` file not writable or URLs don't match exactly | Check that file exists and is writable; verify URL format |
| Training ignores config model | Expected behavior | Training uses GUI model selector, not config |
| Screen capture is slow | Model too large | Use `yolov8n.pt` instead of `yolov8m.pt` |
| StreamCut is slow | `process_workers` too high | Reduce to 2–4 |
| Getting "rate limit" from Twitch | Too many downloads in parallel | Set `max_download_workers: 1` |
| Classes don't match model | `class_map` missing | Run "Show Classes from Model" in GUI |

---

## 📊 **Performance Tuning**

| Bottleneck | Change | Expected Result |
|-----------|--------|-----------------|
| GPU memory (out of memory) | Reduce `batch` size or use `amp: true` | Slower but fits in memory |
| CPU maxed during screen capture | Use smaller model (`yolov8n`) | Fast but less accurate |
| GPU idle during StreamCut | Increase `process_workers` | Better GPU utilization |
| Download too slow | Increase `max_download_workers` (max 3) | Faster but risk rate limit |
| Processing takes hours | Increase `time_interval` | 3x faster, fewer frames |
| Dataset too large | Increase `save_interval` | Fewer images, less storage |

---

## 🔗 **Related**

- [📖 Main README](../README.md) — Full user guide
- [⚙️ View current config.json](../configs/config.json)
- [🎬 View current configStreamCut.json](../configs/configStreamCut.json)

