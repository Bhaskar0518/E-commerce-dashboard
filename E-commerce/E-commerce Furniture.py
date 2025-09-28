import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings("ignore")

# 1. Load Data
df = pd.read_csv(r"C:\Users\vbhas\OneDrive\Desktop\E-commerce\ecommerce_furniture_dataset_2024.csv")
df = df[["productTitle", "price", "sold", "tagText"]]
df.columns = ["product_title", "price", "sold", "shipping_tag"]
df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
df["sold"] = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)

# 2. Simplify Product Categories
def simplify(title):
    title = title.lower()
    if "chair" in title:
        return "Portable Chair"
    elif "book" in title:
        return "Books"
    elif "sofa" in title or "couch" in title or "living" in title:
        return "Living Room"
    elif "table" in title:
        return "Table"
    elif "dresser" in title:
        return "Bedroom"
    elif "patio" in title or "outdoor" in title:
        return "Outdoor"
    else:
        return "Other"

df["product_type"] = df["product_title"].apply(simplify)
df["is_popular"] = (df["sold"] > 50).astype(int)

# 3. Prepare Features
X = pd.get_dummies(df[["product_type", "shipping_tag"]])
X = X.astype(float)  # ✅ Fix for SMOTE compatibility
y = df["is_popular"]

# 4. Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save Model
with open(r"C:\Users\vbhas\OneDrive\Desktop\E-commerce\popularity_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 7. Evaluation
print("\n🔍 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 8. Visualizations
plt.figure(figsize=(8, 4))
sns.histplot(df["price"], bins=20, kde=True)
plt.title("Price Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
top = df.groupby("product_type")["sold"].sum().reset_index().sort_values("sold", ascending=False).head(10)
sns.barplot(x="product_type", y="sold", data=top)
plt.title("Top Selling Categories")
plt.tight_layout()
plt.show()
