# 🛍️ Smart Store Footfall & Behavior Analytics

> **Turning in-store customer movement into actionable retail decisions — powered by the MERL Shopping Dataset.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MERL%20Shopping-0057B7?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-00C9A7?style=flat-square)

---

## 📌 Overview

This project analyzes **customer shopping behavior** at retail shelves using the [MERL Shopping Dataset](https://www.merl.com/demos/merl-shopping-dataset) — a collection of 106 surveillance-style videos featuring 41 subjects across 3 sessions each.

The dashboard tracks and visualizes **5 in-store shopping actions**:

| # | Action | Business Relevance |
|---|--------|--------------------|
| 1 | Reach to Shelf | Product accessibility & placement |
| 2 | Retract from Shelf | Decision abandonment signal |
| 3 | Hand in Shelf | Deep product engagement |
| 4 | Inspect Product | Pick-up conversion indicator |
| 5 | Inspect Shelf | Browse-without-buy behavior |

---

## 🎯 What This Dashboard Does

- **Action Frequency Analysis** — which behaviors dominate customer interactions
- **Duration Analysis** — how long customers engage with each action type
- **Activity Heatmap** — when in the session each action clusters
- **Subject Timeline** — per-customer session breakdown
- **Confidence Score Distribution** — detection reliability across actions
- **Business Insight Engine** — automated recommendations from behavioral patterns

---

## 🧠 Key Business Insights Generated

> *"Customers who Inspect Product at 2× the rate of Inspect Shelf are converting browsers into buyers — good shelf-eye-level alignment."*

> *"Hand in Shelf has the longest dwell time — premium product placement here maximizes exposure."*

> *"Low Retract from Shelf frequency signals confident purchase decisions, not abandonment."*

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-footfall-analysis.git
cd retail-footfall-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

**With simulated demo data (no download needed):**
```bash
streamlit run dashboard.py
```

**With real MERL data:**
1. Download from the [MERL website](https://www.merl.com/demos/merl-shopping-dataset)
2. Place `Results/` folder in the project root
3. Run: `streamlit run dashboard.py`
4. Switch to **"Real MERL Data"** in the sidebar

---

## 📁 Project Structure

```
retail-footfall-analysis/
│
├── dashboard.py                  # 🖥️  Main Streamlit app
├── data_loader.py                # 🔧  Data loading & simulation logic
├── requirements.txt              # 📦  Python dependencies
│
├── Results/                      # 📂  MERL Results folder (not in repo)
│   ├── DetectedActions/
│   │   ├── 1.mat                 #      Reach to Shelf detections
│   │   ├── 2.mat                 #      Retract from Shelf
│   │   ├── 3.mat                 #      Hand in Shelf
│   │   ├── 4.mat                 #      Inspect Product
│   │   └── 5.mat                 #      Inspect Shelf
│   ├── frame_names.txt           #      Frame-to-filename mapping
│   └── framepreds.mat            #      Per-chunk prediction scores
│
└── Labels_MERL_Shopping_Dataset/ # 📂  Ground truth labels (not in repo)
    └── xx_yy_label.mat           #      Per-subject session labels
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Dataset | MERL Shopping Dataset (Mitsubishi Electric, 2016) |

---

## 📊 Dashboard Preview

> The dashboard runs in two modes:
> - **Demo Mode** — uses simulated data mirroring real MERL structure (works immediately, no data download)
> - **Live Mode** — loads real `.mat` detection files from the Results folder

---

## 💼 Business Context

This project mirrors real-world retail intelligence solutions like **Brysk** (camera-based in-store analytics) and similar computer vision platforms used in modern retail.

Instead of just showing code, this project answers:
- **"Where should high-margin products go?"** → Place near high Reach-to-Shelf zones
- **"Which shelf needs redesign?"** → Where Inspect Shelf is high but Inspect Product is low
- **"When are customers most engaged?"** → Activity heatmap by session time

---

## 📚 Dataset Citation

```bibtex
@InProceedings{Singh_2016_CVPR,
  author    = {Singh, Bharat and Marks, Tim K. and Jones, Michael and Tuzel, Oncel and Shao, Ming},
  title     = {A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection},
  booktitle = {CVPR},
  year      = {2016}
}
```

---

## 👩‍💻 Author

**Srijita Kayal**  
MBA — Data Science & Business Analytics | BIBS, Kolkata  
[LinkedIn](https://linkedin.com/in/srijita-kayal-data-analytic-business-analytic) · [GitHub](https://github.com/srijita205) · [Kaggle](https://kaggle.com/srijitakayal)

---

<sub>Dataset © Mitsubishi Electric Research Laboratories (MERL), 2016. Used for educational and research purposes.</sub>
