# 🧠 DICOM Processing & Visualization Toolkit

---

## 🔍 DICOM Processing & Visualization

- **Load and process DICOM CT series**
- **Maximum Intensity Projection (MIP)** visualization across axial, coronal, and sagittal planes
- **Tumor and liver segmentation overlay** with alpha blending
- **Rotating MIP animations** with smooth coronal-sagittal transitions

---

## 📊 Advanced Visualization

- **Interactive 3D volume rendering**
- **Alpha-fused segmentation overlays**
- **Histogram equalization (CLAHE)** for enhanced contrast
- **CT windowing** for optimal tissue visualization
- **Animated GIF generation** with customizable rotation angles

---

## 🎯 3D Rigid Coregistration

- **Automated image registration** using Sum of Squared Differences (SSD)
- **Liver mask preservation** and alignment assessment
- **Numerical quality metrics**: NCC, MSE, RMSE, MAE, Dice coefficient
- **Visual assessment** with before/after comparisons
- **Convergence monitoring** and optimization tracking

---

## ⚙️ Installation

### Requirements

```bash
pip install numpy matplotlib scipy scikit-image pydicom pyyaml tqdm

