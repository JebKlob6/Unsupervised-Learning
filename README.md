
**Consolidated Experiment Runner for Comprehensive Unsupervised Learning Analysis**

This repository contains a complete implementation of unsupervised learning experiments including clustering,
dimensionality reduction, and neural network analysis. The code eliminates duplication and systematically captures all
required metrics for academic analysis.

## 🚀 Features

### **Core Experiments**

- **Intelligent Clustering**: K-Means and EM with automatic optimal cluster selection using multiple validation methods
- **Baseline Clustering**: Advanced cluster validation including elbow method, silhouette analysis, gap statistic, and
  information criteria
- **Dimensionality Reduction**: PCA, ICA, and Random Projection with comprehensive diagnostics
- **DR + Clustering**: All 12 combinations of 3 DR methods × 2 clustering algorithms × 2 datasets with optimal cluster
  selection
- **Neural Network Analysis**: Performance comparison with learning curves and visualizations

### **Intelligent Cluster Selection**

- **Elbow Method**: Automatic detection of optimal K using rate of change analysis
- **Silhouette Analysis**: Maximization of average silhouette score across cluster range
- **Gap Statistic**: Comparison with null reference distribution for robust cluster number selection
- **Information Criteria**: BIC/AIC minimization for EM component selection
- **Consensus Method**: Multi-method voting system for robust optimal cluster determination

### **Comprehensive Metrics Collection**
- **Clustering Metrics**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
- **External Validation**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- **EM-Specific**: BIC, AIC, Log-Likelihood scores
- **DR Diagnostics**: Explained variance, kurtosis analysis, reconstruction error
- **Neural Network**: Accuracy, training time, learning curves

### **Advanced Visualizations**

- **Cluster Selection Diagnostics**: Multi-method cluster validation plots showing elbow curves, silhouette scores, gap
  statistics, and information criteria
- **PCA**: Dual subplot with scree plot and cumulative variance (95% threshold line)
- **ICA**: Kurtosis analysis with component plots and distribution histograms
- **Random Projection**: Component variance and reconstruction error analysis
- **Neural Networks**: 3-subplot grid with learning curves, loss curves, and performance summary

## 🔧 Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git access to the repository
- Terminal/Command Prompt access


### 2. Set up Python Environment
#### **Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python --version
pip list
```

#### **Windows (Command Prompt):**

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python --version
pip list
```

#### **Windows (PowerShell):**

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python --version
pip list
```

### 3. Troubleshooting Platform-Specific Issues

#### **Linux:**

```bash
# If python3 command not found, try:
sudo apt update && sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
sudo yum install python3 python3-pip                                  # CentOS/RHEL
sudo pacman -S python python-pip                                      # Arch Linux

# If permission issues with pip:
pip install --user --upgrade pip
pip install --user -r requirements.txt
```

#### **macOS:**

```bash
# If python3 not found, install via Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python

# Alternative: Install via Python.org or pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

#### **Windows:**

```cmd
# If python command not found:
# 1. Download Python from https://python.org
# 2. Ensure "Add Python to PATH" is checked during installation
# 3. Restart Command Prompt

# If execution policy issues in PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📊 Dataset Information

The experiments run on two datasets located in the `/data` directory:

1. **Company Bankruptcy Dataset** (`company_bankruptcy_data.csv`)
    - 6,819 samples with 95 numerical features
    - Binary classification: Bankrupt vs Non-Bankrupt

2. **Global Cancer Patients Dataset** (`global_cancer_patients_2015_2024.csv`)
    - Multi-class cancer stage classification
    - Mixed categorical and numerical features

## 🎯 Running Experiments

### **Quick Start**

#### **Linux/macOS:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all experiments
python run_experiments.py
```

#### **Windows:**

```cmd
# Activate virtual environment (Command Prompt)
.venv\Scripts\activate

# OR for PowerShell:
# .venv\Scripts\Activate.ps1

# Run all experiments
python run_experiments.py
```

### **What Happens**
The consolidated experiment runner will:

1. **Load and preprocess** both datasets with automatic feature engineering
2. **Run baseline clustering** (K-Means and EM) on original data
3. **Apply dimensionality reduction** (PCA, ICA, RP) with multiple component counts
4. **Generate comprehensive visualizations** for each DR method
5. **Run clustering on reduced data** (12 combinations total)
6. **Perform neural network analysis** with learning curve generation
7. **Save all results** and generate summary reports

### **Expected Runtime**
- Complete analysis: ~5-15 minutes depending on system performance
- Progress indicators show real-time status
- All results automatically saved with timestamps

## 📁 Output Structure

### **Figures Directory (`/figures`)**
```
figures/
├── pca/              # PCA scree plots and cumulative variance
├── ica/              # ICA kurtosis analysis and distributions  
├── rp/               # Random Projection variance and reconstruction error
├── nndr/             # Neural Network analysis (3-subplot grids)
├── consolidated/     # Summary visualizations and cluster selection diagnostics
└── consolidated_results/ # Final combined results
```

### **Key Visualizations**

**Cluster Selection Diagnostics**

- `{DatasetName}_{algorithm}_{dr_method}_{components}_cluster_selection.png`: Multi-method validation plots showing:
    - Elbow method inertia curves with optimal K highlighted
    - Silhouette score optimization across cluster range
    - Gap statistic analysis with reference distributions
    - BIC/AIC information criteria for EM component selection

**PCA Analysis**
- `{DatasetName}_PCA_{components}.png`: Dual subplot with scree plot and cumulative variance

**ICA Analysis**

- `{DatasetName}_ICA_{components}.png`: Kurtosis plots and distribution histograms

**Random Projection Analysis**
- `{DatasetName}_RP_{components}.png`: Component variance and reconstruction error

**Neural Network Analysis**
- `{DatasetName}_{dr_method}_{components}_NN.png`: 3-subplot grid with:
    - Learning curves (training vs validation accuracy)
    - Training loss over iterations
    - Performance summary bar chart

## 🔬 Technical Implementation

### **Code Architecture**
- **Single consolidated file**: `run_experiments.py` (eliminates code duplication)
- **Modular design**: Separate methods for each experiment type
- **Comprehensive error handling**: Graceful failure with detailed logging
- **Reproducible results**: Fixed random seeds throughout

### **Key Classes**

- `OptimalClusterSelector`: Intelligent cluster number selection using multiple validation methods
- `ConsolidatedExperiments`: Main experiment runner with automatic cluster optimization
- `SimpleResultsManager`: Results storage and reporting

### **Data Processing Pipeline**
1. Automatic feature type detection (categorical vs numerical)
2. StandardScaler for numerical features
3. OneHotEncoder for categorical features
4. Train/test stratified splitting
5. Proper handling of sparse matrices

### **Optimal Cluster Selection Details**

- **Multi-Method Approach**: Each clustering task uses 3-5 different validation methods simultaneously
- **Consensus Algorithm**: Results are combined using mode/median consensus for robust selection
- **Adaptive Range**: Automatically determines reasonable cluster range based on data size
- **Comprehensive Diagnostics**: All validation curves and scores are visualized for interpretation
- **Fallback Strategies**: Graceful handling of method failures with intelligent defaults

### **Dimensionality Reduction Details**
- **PCA**: Captures explained variance ratios with 95% threshold analysis
- **ICA**: FastICA with deflation algorithm and kurtosis maximization
- **RP**: Gaussian Random Projection with reconstruction error analysis

## 📈 Experiment Results

### **Baseline Clustering Performance**
The system automatically determines optimal cluster counts:
- **Cancer Dataset**: Tests k ∈ {2, 3, 5} clusters
- **Bankruptcy Dataset**: Tests k ∈ {2, 3, 5} clusters

### **Dimensionality Reduction Analysis**
Component counts tested: {5, 10, 15} (automatically adjusted based on feature count)

### **Neural Network Configuration**
- Architecture: 2 hidden layers (64, 32 neurons)
- Activation: ReLU
- Max iterations: 200
- Learning rate: 0.001
- **Analysis**: Only performed on Cancer dataset as specified

## 🧪 Dependencies

```
scikit-learn>=1.4.2    # Core ML algorithms
pandas>=2.2.2          # Data manipulation
numpy>=1.26.4          # Numerical computing
matplotlib>=3.8.4      # Plotting
scipy>=1.12.0          # Statistical functions
```

### **Platform Compatibility**

- ✅ **Linux**: Fully tested on Ubuntu 20.04+, CentOS 7+, Arch Linux
- ✅ **macOS**: Compatible with macOS 10.15+ (Intel and Apple Silicon)
- ✅ **Windows**: Tested on Windows 10/11 with Python 3.10+

### **Performance Notes**

- **Linux**: Optimal performance with native compilation support
- **macOS (Apple Silicon)**: Excellent performance with optimized NumPy/SciPy
- **Windows**: May require Visual C++ Build Tools for some scientific packages

### **Memory Requirements**

- **Minimum**: 8GB RAM (reduced dataset processing)
- **Recommended**: 16GB+ RAM (full dataset analysis)
- **Storage**: ~2GB free space for results and figures

## 🔧 Cross-Platform Troubleshooting

### **Common Issues & Solutions**

#### **Issue: "python3: command not found" (Linux)**

```bash
# Solution: Install Python 3
sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
sudo yum install python3 python3-pip               # CentOS/RHEL
sudo pacman -S python python-pip                   # Arch Linux
```

#### **Issue: Permission denied errors (Linux/macOS)**

```bash
# Solution: Use user installation
pip install --user -r requirements.txt

# OR: Fix virtual environment permissions
sudo chown -R $USER:$USER .venv/
```

#### **Issue: Matplotlib backend errors (Linux headless)**

```bash
# Solution: Install GUI backend or use Agg backend
sudo apt install python3-tk  # Ubuntu/Debian
export MPLBACKEND=Agg         # Force non-interactive backend
```

#### **Issue: Memory errors during execution**

```bash
# Solution: Monitor and adjust based on system
# Linux/macOS:
htop  # Monitor memory usage

# Windows:
# Use Task Manager to monitor memory usage
# Consider reducing dataset size or component counts in script
```

#### **Issue: Slow execution on older systems**

```bash
# Solution: Reduce computational load
# Edit run_experiments.py:
# - Reduce component_counts = [5] instead of [5, 10, 15]
# - Use smaller max_k in cluster selection
# - Reduce hyperparameter grid size for NN
```

### **Development Environment Setup**

```bash
# For development across platforms:
git clone <repository>
cd unsupervised-learning

# Linux/macOS development setup:
python3 -m venv dev-env
source dev-env/bin/activate
pip install -r requirements.txt
pip install jupyter ipython  # Optional for interactive development

# Windows development setup:
python -m venv dev-env
dev-env\Scripts\activate
pip install -r requirements.txt
pip install jupyter ipython  # Optional for interactive development
```

### **File Path Compatibility**

The code automatically handles cross-platform file paths using Python's `os.path` and pathlib. All file operations work
identically across:

- **Linux**: `/home/user/project/data/file.csv`
- **macOS**: `/Users/user/project/data/file.csv`
- **Windows**: `C:\Users\user\project\data\file.csv`

### **Performance Optimization by Platform**

```bash
# Linux: Enable multi-threading optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# macOS: Optimize for Apple Silicon
export OPENBLAS_NUM_THREADS=4

# Windows: Set processor affinity (PowerShell)
$proc = Get-Process -Name python
$proc.ProcessorAffinity = 15  # Use first 4 cores
```
