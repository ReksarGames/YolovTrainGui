# 📝 YDS Documentation Changelog

## v1.2.0 - Documentation Reorganization (2025-01-28)

### 📚 Structure Improvements
- **Moved documentation to `/docs` folder:**
  - `README_ru.md` → `docs/README_ru.md`
  - `HELP.md` → `docs/HELP.md`
  - New: `docs/HELP_ru.md` (Russian configuration guide)

- **Updated all cross-references:**
  - Links in `README.md` now point to `docs/` folder
  - Image paths corrected for new structure
  - Navigation between languages simplified

### ✨ Documentation Enhancements

#### README.md (English)
- Added language selector badges (English/Русский)
- Simplified installation with `setup.bat` quick start
- Collapsed advanced sections in `<details>` tags:
  - Configuration details
  - Keyboard shortcuts
  - Model downloading and management
- Removed performance benchmarks (moved to separate reference)
- Streamlined workflow visualization (4 main steps)
- Clarified StreamCut purpose: "mines thousands of labeled images from Twitch streams without manual work"

#### README_ru.md (Russian)
- Complete Russian translation of updated README.md
- Parallel structure with English version
- Moved to `/docs` for better organization

#### HELP.md (English)
- **Completely rewritten for clarity:**
  - Reduced from 789 lines to ~350 lines
  - Focus on practical configuration
  - Removed redundant information
  
- **Key sections:**
  - `config.json` parameter reference with tables
  - `configStreamCut.json` worker configuration guide
  - Critical worker tuning:
    - `max_download_workers`: 2-3 (avoid Twitch ban)
    - `split_workers`: CPU core count
    - `process_workers`: GPU thread count
  - Recommended presets (Fast, High Quality, Weak GPU, Strong GPU)
  - Common configuration issues and solutions
  - Tips for different hardware scenarios

#### HELP_ru.md (NEW - Russian)
- Complete Russian translation of HELP.md
- Same professional structure and content
- All examples and presets translated
- Worker tuning guidelines translated

### 🔧 Configuration & Setup

#### setup.bat (NEW - Windows)
- Automated installation script for Windows users
- Features:
  - Python 3.8+ detection
  - Virtual environment creation
  - Automatic dependency installation
  - Installation verification
  - Clear error messages

### 🗑️ Removed Content
- CLI examples from ONNX Benchmarking (GUI provides same functionality)
- Individual performance benchmarks (RTX 3080 specific)
- GPU acceleration setup section (complex, rarely needed)
- Installation troubleshooting (moved to FAQ in README)

### 🎯 Focus Areas
- **Simplified Installation:** One-command setup with `setup.bat`
- **Clear Workflow:** 4-step process from data collection to training
- **GUI-First Approach:** Minimized CLI documentation
- **Worker Configuration:** Detailed guidance for StreamCut optimization
- **Bilingual Support:** English + Russian documentation
- **Professional Structure:** Clean navigation, collapsed advanced content

### 📊 Current Documentation Structure
```
YolovTrainGui/
├── README.md                     # Primary English guide
├── setup.bat                     # Windows automated setup
└── docs/
    ├── README_ru.md              # Russian workflow guide
    ├── HELP.md                   # English configuration reference
    ├── HELP_ru.md                # Russian configuration reference
    ├── CHANGELOG.md              # This file
    └── images/
        ├── yds/
        │   ├── training.PNG
        │   └── dataset.PNG
        └── streamcut/
            └── streamcut.PNG
```

### 🚀 Next Steps for Users
1. **New Users:** Read `README.md`, then use `setup.bat` to install
2. **Configuration Needs:** Check `docs/HELP.md` or `docs/HELP_ru.md`
3. **Russian Users:** Start with `docs/README_ru.md`
4. **Advanced Tuning:** See "Worker Configuration" in HELP files

### 📞 Support Resources
- Configuration issues → Check `docs/HELP.md` troubleshooting section
- Installation problems → See `setup.bat` error messages
- Features → Check workflow steps in `README.md`
- Russian support → See `docs/README_ru.md` and `docs/HELP_ru.md`

---

**Note:** All documentation has been reorganized for better maintainability and user experience. Links and file structure have been updated accordingly.
