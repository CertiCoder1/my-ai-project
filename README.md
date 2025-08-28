# Iris Classifier Project

This project trains and evaluates **Decision Tree** and **k-Nearest Neighbors (k-NN)** classifiers on the Iris dataset.  
It includes a reproducible CLI training script and generates saved models + confusion matrix plots.

---

## 📂 Repository Structure

# Choose or make a parent folder, then:
mkdir iris-classifier
cd iris-classifier
python -m venv venv          # optional but recommended
source venv/bin/activate     # macOS/Linux
# .venvScriptsactivate     # Windows PowerShell
pip install --upgrade pip
pip install scikit-learn matplotlib seaborn jupyter



iris-classifier/
├── data/ # (empty – Iris is loaded from scikit‑learn)
├── notebooks/
│ └── iris_model.ipynb # walk‑through notebook
├── src/
│ └── train.py # reproducible CLI script
├── models/ # created automatically (model & figures)
├── .gitignore
├── README.md
└── requirements.txt


# Train both models
python -m src.train --model both

# Train only Decision Tree with max depth of 5
python -m src.train --model dt --max-depth 5

# Train only k-NN with 7 neighbors
python -m src.train --model knn --n-neighbors 7



Command-line arguments
	•	--model : dt, knn, or both (default: both)
	•	--test-size : Test set proportion (default: 0.2)
	•	--random-state : Random seed (default: 42)
	•	--max-depth : Max depth for Decision Tree (default: None)
	•	--n-neighbors : Number of neighbors for k-NN (default: 5)
	•	--outputs-dir : Where to save outputs (default: outputs/)



