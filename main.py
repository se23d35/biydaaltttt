import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. ӨГӨГДӨЛ ЦУГЛУУЛАХ БА БОЛОВСРУУЛАЛТ
# ==========================================
# Жишээ: Зургуудаас гаргаж авсан CSV эсвэл бэлэн дата
# Хэрэв CSV байхгүй бол туршилтаар өгөгдөл үүсгэе
try:
    df = pd.read_csv('driver_data.csv')
except:
    # Туршилтын өгөгдөл үүсгэх (нүдний харьцаа, толгойн өнцөг, статус)
    data = {
        'eye_ratio': np.random.normal(0.3, 0.05, 100).tolist() + np.random.normal(0.1, 0.05, 100).tolist(),
        'head_angle': np.random.normal(0, 10, 100).tolist() + np.random.normal(45, 15, 100).tolist(),
        'status': [1]*100 + [0]*100 # 1: Focused, 0: Distracted
    }
    df = pd.DataFrame(data)

# NaN утгыг шийдэх
df.fillna(df.mean(), inplace=True)

# Outlier шалгах (Boxplot)
plt.figure(figsize=(10, 4))
sns.boxplot(data=df)
plt.title("Outlier Detection (Preprocessing)")
plt.show()

# Өгөгдөл хуваах (Hold-out 80/20)
X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (KNN, SVC-д заавал хэрэгтэй)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. МОДЕЛЬ СУРГАХ БА ХАРЬЦУУЛАЛТ
# ==========================================

# А. SVC - Hyperparameter tuning (Grid Search)
param_svc = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
grid_svc = GridSearchCV(SVC(), param_svc, cv=5)
grid_svc.fit(X_train_scaled, y_train)

# Б. Decision Tree - Hyperparameter tuning
param_dt = {'max_depth': [3, 5, 10, None], 'criterion': ['gini', 'entropy']}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_dt, cv=5)
grid_dt.fit(X_train, y_train)

# ==========================================
# 3. ҮНЭЛГЭЭ БА ШИНЖИЛГЭЭ
# ==========================================
def evaluate(model, X_t, y_t, name, is_scaled=False):
    pred = model.predict(X_t)
    print(f"\n--- {name} Report ---")
    print(classification_report(y_t, pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_t, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    return accuracy_score(y_t, pred)

acc_svc = evaluate(grid_svc.best_estimator_, X_test_scaled, y_test, "SVC (Standardized)")
acc_dt = evaluate(grid_dt.best_estimator_, X_test, y_test, "Decision Tree")

# Аль нь илүү вэ?
best_model = grid_svc.best_estimator_ if acc_svc > acc_dt else grid_dt.best_estimator_
print(f"\nХамгийн өндөр үр дүн: {max(acc_svc, acc_dt):.2f}")

# ==========================================
# 4. БҮТЭЭГДЭХҮҮН БОЛГОХ (.pkl)
# ==========================================
joblib.dump(best_model, 'best_driver_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Загварыг .pkl хэлбэрээр хадгаллаа.")

# ==========================================
# 5. ИНТЕРФЭЙС / REAL-TIME ДЕТЕКШН
# ==========================================
def start_interface():
    # Энд өмнө нь бичсэн Камерын кодыг дуудаж болно
    # Моделио ачаалах: loaded_model = joblib.load('best_driver_model.pkl')
    print("Интерфэйс бэлэн боллоо. (Камер идэвхжихэд бэлэн)")

start_interface()
