import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\vbhas\OneDrive\Desktop\E-commerce\ecommerce_furniture_dataset_2024.csv")
df = df[["productTitle", "price", "sold", "tagText"]]
df.columns = ["product_title", "price", "sold", "shipping_tag"]
df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
df["sold"] = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)

# Simplify product types
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

# Load model
with open(r"C:\Users\vbhas\OneDrive\Desktop\E-commerce\popularity_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare feature columns
features = pd.get_dummies(df[["product_type", "shipping_tag"]])
feature_columns = features.columns.tolist()

# Streamlit UI
st.set_page_config(page_title="Furniture Popularity Predictor", layout="wide")
st.title("🪑 E-commerce Furniture Dashboard")

# Top Sellers
st.header("📊 Top Selling Categories")
st.caption("Quick view of best-selling product types.")
top = df.groupby("product_type")["sold"].sum().reset_index().sort_values("sold", ascending=False).head(10)
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x="product_type", y="sold", data=top, ax=ax1)
plt.xticks(rotation=45, ha="right")
st.pyplot(fig1)

# Price Distribution
st.header("💰 Price Spread")
st.caption("Distribution of product prices.")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df["price"], bins=20, kde=True, ax=ax2)
st.pyplot(fig2)

# Prediction
st.header("🔮 Predict Popularity")
st.caption("Estimate if a product will be popular.")
product_type = st.selectbox("Product Type", sorted(df["product_type"].unique()))
shipping_tag = st.selectbox("Shipping Tag", sorted(df["shipping_tag"].astype(str).unique()))

if st.button("Predict Popularity"):
    input_data = pd.DataFrame([{
        "product_type": product_type,
        "shipping_tag": shipping_tag
    }])
    input_encoded = pd.get_dummies(input_data).reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    result = "Popular ✅" if prediction == 1 else "Not Popular ❌"
    st.success(f"Prediction: {result}")
