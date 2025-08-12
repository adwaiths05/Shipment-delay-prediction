# 📦 Shipment Delay Prediction with Explainable AI

Predict shipment delays in US logistics operations using a machine learning pipeline built on **LightGBM** with **SHAP explainability**.  
This project uses a synthetic dataset simulating real-world logistics operations, including missing values, outliers, and seasonal effects.

---

## 🚀 Project Overview
Logistics companies need to proactively detect potential delays to optimize routing, choose the right carriers, and reduce operational costs.  
This project trains a machine learning model to:
- Predict **whether a shipment will be delayed**
- Provide **feature importance and explanations** for each prediction

---

### Features:
- `Shipment_ID` — Unique shipment identifier  
- `Origin_Warehouse` — Origin warehouse location  
- `Destination` — Destination location  
- `Carrier` — Shipping carrier  
- `Shipment_Date` — Date the shipment left the warehouse  
- `Delivery_Date` — Actual delivery date  
- `Weight_kg` — Shipment weight  
- `Cost` — Shipping cost in USD  
- `Status` — Delivery status  
- `Distance_miles` — Distance from origin to destination  
- `Transit_Days` — Planned transit duration

### Target:
- **`is_delayed`** — Binary label:  
  - `1` = Delivery was late  
  - `0` = Delivered on time

---

## 🛠 Tech Stack
- **Python** — Core language  
- **Pandas, NumPy** — Data preprocessing  
- **LightGBM** — Model training  
- **scikit-learn** — Data splitting, metrics  
- **SHAP** — Explainability  
- **Matplotlib** — Visualizations  
- **Jupyter Notebook** — Development environment

---

## 🔍 Key Steps
1. **Data Cleaning & Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Extract temporal features (month, day of week)
   - Create binary target `is_delayed`
2. **Model Training**
   - LightGBM classifier
   - Class weight balancing for imbalanced dataset
3. **Evaluation**
   - Classification report (precision, recall, F1)
   - ROC-AUC score
4. **Explainability**
   - Global SHAP summary plots
   - Local force plots for individual predictions

## ⚡ Quick Start
**1. Clone the repository**
```bash
git clone https://github.com/adwaiths05/shipment-delay-prediction.git
cd shipment-delay-prediction
```
**2. Install dependencies**

```bash
pip install -r requirements.txt
```
**3. Run Jupyter Notebook**

```bash
jupyter notebook notebooks/shipment_delay_prediction.ipynb
```
