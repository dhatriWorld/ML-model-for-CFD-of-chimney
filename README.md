# 🌪️ Machine Learning Model for CFD of Chimney

This project applies **Machine Learning (ML)** techniques to predict **Computational Fluid Dynamics (CFD)** simulation results for a chimney system.  
The dataset consists of CFD simulation outputs for varying **Inlet Temperature** and **Flux** conditions. The models learn to predict key flow field variables such as:

- Pressure  
- Velocity Magnitude  
- Total Temperature  
- Total Energy  

This approach helps reduce the computational cost of CFD by providing quick approximations using ML models.

---

## ⚙️ Features

- **Exploratory Data Analysis (EDA)**  
  - Scatter plots of Flux vs Pressure  
  - Correlation heatmap of input/output features  
  - 3D visualization of velocity variation along the chimney  

- **Machine Learning Models**  
  - XGBoost Regressor ✅ (primary model, saved for predictions)  
  - K-Nearest Neighbors Regressor  
  - Random Forest Regressor  
  - Artificial Neural Network (ANN) using TensorFlow/Keras  

- **Prediction Functions**  
  - Single-cell prediction (for a given inlet temperature, flux, and cell coordinates)  
  - Full-chimney prediction (for an entire CFD domain at chosen conditions)  

---

## 📊 Input Features

- `X_Coordinate`  
- `Y_Coordinate`  
- `Inlet_Temperature`  
- `Flux`

## 🎯 Predicted Outputs

- `Pressure`  
- `Velocity_Magnitude`  
- `Total_Temperature`  
- `Total_energy`

---

## 🚀 How to Run

### 1️⃣ Clone this repository
```bash
git clone https://github.com/yourusername/ML-CFD-Chimney.git
cd ML-CFD-Chimney
2️⃣ Install dependencies

pip install -r requirements.txt
Required packages:
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
tqdm
joblib

3️⃣ Prepare the dataset

Place CFD CSV files inside the data/ folder.
File format should be like:
283K_400.csv
293K_600.csv
...
4️⃣ Run the script / notebook
python ML-CFD-Chimney.py
📌 Example Usage
🔹 Single Cell Prediction
pred_df = making_predictions(inlet_temp=293, flux=600, 
                             x_coordinate=0.1, y_coordinate=0.2)
print(pred_df)
🔹 Full Chimney Prediction
pred_df = full_prediction(inlet_temp=293, flux=600)
print(pred_df.head())

📈 Results
Root Mean Squared Error (RMSE) is calculated for all models.
XGBoost gave the best performance and is used for saving and inference.
ANN was also tested for non-linear behavior capture.

🏗️ Future Improvements
Add hyperparameter tuning for XGBoost & Random Forest
Explore Physics-Informed ML models for CFD
Build a web dashboard for real-time predictions
Extend dataset with more operating conditions

