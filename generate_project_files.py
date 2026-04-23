import pandas as pd
import numpy as np
import nbformat as nbf
import warnings

# 1. Generate Weather Dataset
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(25, 10, n_samples)
humidity = np.random.normal(60, 20, n_samples)
wind_speed = np.random.normal(15, 8, n_samples)
pressure = np.random.normal(1013, 10, n_samples)

# Simple rule to define weather condition: 'Sunny', 'Rainy', 'Cloudy'
conditions = []
for i in range(n_samples):
    if humidity[i] > 75 and pressure[i] < 1010:
        conditions.append('Rainy')
    elif humidity[i] > 60 or wind_speed[i] > 20:
        conditions.append('Cloudy')
    else:
        conditions.append('Sunny')

df = pd.DataFrame({
    'Temperature_C': temperature,
    'Humidity_%': humidity,
    'Wind_Speed_kmh': wind_speed,
    'Pressure_hPa': pressure,
    'Condition': conditions
})
df.to_csv('weather_dataset.csv', index=False)
print("Weather dataset created successfully.")

# 2. Generate Jupyter Notebook
nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Ensemble Learning Techniques\n\nThis notebook demonstrates Ensemble Learning Techniques using a synthetic Weather dataset."),
    
    nbf.v4.new_markdown_cell("## 1. Import Libraries"),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport joblib\nimport warnings\nwarnings.filterwarnings('ignore')"),

    nbf.v4.new_markdown_cell("## 2. Load and Preprocess Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('weather_dataset.csv')\ndisplay(df.head())\n\n# Encode target\nle = LabelEncoder()\ndf['Condition_Encoded'] = le.fit_transform(df['Condition'])\n\nX = df[['Temperature_C', 'Humidity_%', 'Wind_Speed_kmh', 'Pressure_hPa']]\ny = df['Condition_Encoded']\n\n# Train test split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Scaling\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)"),

    nbf.v4.new_markdown_cell("## 3. Implement Random Forest Algorithm"),
    nbf.v4.new_code_cell("rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\nrf_model.fit(X_train_scaled, y_train)\ny_pred_rf = rf_model.predict(X_test_scaled)"),

    nbf.v4.new_markdown_cell("## 4. Evaluate Model Performance"),
    nbf.v4.new_code_cell("accuracy = accuracy_score(y_test, y_pred_rf)\nprint(f'Random Forest Accuracy: {accuracy:.4f}')\nprint('\\nClassification Report:')\nprint(classification_report(y_test, y_pred_rf, target_names=le.classes_))\n\n# Confusion Matrix\ncm = confusion_matrix(y_test, y_pred_rf)\nplt.figure(figsize=(6,4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)\nplt.title('Random Forest Confusion Matrix')\nplt.xlabel('Predicted')\nplt.ylabel('Actual')\nplt.show()"),

    nbf.v4.new_markdown_cell("## 5. Compare with Other Algorithms"),
    nbf.v4.new_code_cell("# Decision Tree\ndt_model = DecisionTreeClassifier(random_state=42)\ndt_model.fit(X_train_scaled, y_train)\ny_pred_dt = dt_model.predict(X_test_scaled)\nprint(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}')\n\n# Logistic Regression\nlr_model = LogisticRegression(random_state=42)\nlr_model.fit(X_train_scaled, y_train)\ny_pred_lr = lr_model.predict(X_test_scaled)\nprint(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')"),

    nbf.v4.new_markdown_cell("## 6. Save the Model"),
    nbf.v4.new_code_cell("joblib.dump(rf_model, 'rf_model.pkl')\njoblib.dump(scaler, 'scaler.pkl')\njoblib.dump(le, 'label_encoder.pkl')\nprint('Models saved successfully!')")
]

with open('Ensemble_Learning_Techniques.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully.")
