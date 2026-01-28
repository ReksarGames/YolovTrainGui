# 🤝 Contributing to YDS

Thank you for your interest in contributing to **YDS (YOLO Dataset Studio)**! We welcome contributions from the community.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of PyQt5 (for GUI contributions)
- Understanding of YOLO framework (for model/training contributions)

### Fork & Clone
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/YolovTrainGui.git
cd YolovTrainGui
```

## Development Setup

### 1. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

### 3. Verify Installation
```bash
python GUI.py  # Should launch the GUI
```

## Making Changes

### Code Style Guidelines

#### Python
- Follow **PEP 8** conventions
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Add type hints where possible
- Document functions with docstrings

Example:
```python
def process_dataset(dataset_path: str, output_dir: str) -> bool:
    """
    Process dataset files in the given directory.
    
    Args:
        dataset_path: Path to input dataset folder
        output_dir: Path to save processed files
        
    Returns:
        True if processing succeeded, False otherwise
    """
    # Implementation
    pass
```

#### GUI (PyQt5)
- Keep components modular
- Separate logic from UI
- Use signals/slots for communication
- Add meaningful window titles and tooltips
- Test UI responsiveness with large datasets

### Commit Message Format

Use clear, descriptive commit messages:
```
[CATEGORY] Brief description (50 chars max)

Detailed explanation of changes (if needed).
- Point 1
- Point 2

Fixes #123
```

**Categories:**
- `[FEATURE]` - New functionality
- `[FIX]` - Bug fix
- `[DOCS]` - Documentation
- `[REFACTOR]` - Code restructuring
- `[TEST]` - Tests added/updated
- `[PERF]` - Performance improvement

Example:
```
[FEATURE] Add batch label export to CSV

Implements export functionality for labeled datasets:
- Supports multiple label formats
- Preserves bounding box precision
- Includes validation checks

Fixes #234
```

### Testing

#### Manual Testing
1. Test with different dataset sizes (small, large)
2. Test on different hardware (CPU, GPU, weak GPU)
3. Verify all UI buttons and workflows
4. Test error handling (missing files, invalid configs)

#### Unit Tests (if applicable)
```bash
python -m pytest tests/
```

## Pull Request Process

### Before Submitting
1. ✅ Update `README.md` if you added features
2. ✅ Update configuration documentation if you changed configs
3. ✅ Test with `python GUI.py`
4. ✅ Check that your code follows PEP 8
5. ✅ Add any new dependencies to `requirements.txt`

### Submitting a PR
1. Push to your fork: `git push origin feature-name`
2. Open a Pull Request on GitHub
3. Fill out the PR template completely:
   - Clear title describing the change
   - Description of changes made
   - Link to related issues (if any)
   - Testing verification
   - Screenshots (for UI changes)

### PR Review Process
- We'll review within 2-3 days
- Provide constructive feedback
- Request changes if needed
- Merge when approved ✅

## Reporting Issues

### Bug Reports
Include:
- **OS & Python version**: `python --version`
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Screenshots/logs** (if applicable)
- **Affected version** (if known)

### Feature Requests
Include:
- **Clear description** of desired functionality
- **Use cases** (why is this needed?)
- **Proposed implementation** (if you have ideas)
- **Alternative solutions** considered

### Template
```markdown
**Description:**
Brief description of bug/feature

**Reproduction Steps:**
1. Step one
2. Step two
3. ...

**Expected Result:**
What should happen

**Actual Result:**
What actually happens

**Environment:**
- OS: Windows 10/11 or Ubuntu 22.04
- Python: 3.10.x
- YDS Version: v1.0.0
```

## Documentation

### Updating Docs
- Keep README.md current with new features
- Update HELP.md for configuration changes
- Maintain both English and Russian versions
- Add examples for new functionality

### Documentation Files
- **README.md** - Main guide, quick start
- **docs/README_ru.md** - Russian translation
- **docs/HELP.md** - Configuration reference
- **docs/HELP_ru.md** - Russian configuration
- **docs/CHANGELOG.md** - Changes history

### Translation
- English version is the source
- Translate to Russian as needed
- Keep formatting and structure identical
- Use consistent terminology

## 💡 Development Tips

### Debugging
```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.error("Error message")
```

### Working with Config Files
- Test with `configs/config.json`
- Test with `configs/configStreamCut.json`
- Validate JSON structure before saving
- Don't hardcode paths - use config

### Performance Considerations
- Heavy operations should run in threads
- Use progress bars for long tasks
- Cache YOLO model to avoid reloading
- Test with large datasets (10k+ images)

### File Structure
```
YolovTrainGui/
├── GUI.py                    # Main entry point
├── Core/
│   ├── train.py             # Training logic
│   ├── StreamCut.py         # VOD processing
│   └── ...
├── configs/
│   ├── config.json          # Main config
│   └── configStreamCut.json # StreamCut config
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Questions?

- 💬 Open an issue for questions
- 📖 Check existing documentation
- 🔍 Search closed issues/PRs
- 📧 Contact maintainers if needed

---

## Code of Conduct

### Be Respectful
- Treat everyone with courtesy
- Welcome diverse perspectives
- Listen to feedback constructively
- Help other contributors

### Be Professional
- Focus on issues and code
- Avoid off-topic discussions
- Report problems appropriately
- Maintain confidentiality when needed

**We're grateful for your contributions! 🙏**

---

**Last Updated:** January 2026
**Maintained by:** YDS Development Team
