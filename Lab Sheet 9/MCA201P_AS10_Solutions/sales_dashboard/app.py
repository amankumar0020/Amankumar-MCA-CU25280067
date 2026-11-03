"""
Sales Data Analysis Dashboard
Run: streamlit run app.py
Dataset: sales_data.csv (sample/demo data)
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sales Analysis", layout="wide")
st.title("Sales Data Analysis Dashboard")
st.caption("Monthly sales, profit, customer segmentation, product performance and regional analysis.")

@st.cache_data
def load_data():
    return pd.read_csv("sales_data.csv", parse_dates=["order_date"])

df = load_data().drop_duplicates()
# Handle missing values and standardize numeric columns.
for col in ["sales", "profit", "quantity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
for col in ["region", "category", "customer_segment", "product"]:
    df[col] = df[col].fillna("Unknown").astype(str)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date"])

regions = sorted(df["region"].unique())
chosen_regions = st.sidebar.multiselect("Region", regions, default=regions)
segments = sorted(df["customer_segment"].unique())
chosen_segments = st.sidebar.multiselect("Customer segment", segments, default=segments)
data = df[df["region"].isin(chosen_regions) & df["customer_segment"].isin(chosen_segments)].copy()
if data.empty:
    st.warning("Choose at least one region and one customer segment.")
    st.stop()

data["month"] = data["order_date"].dt.to_period("M").astype(str)
a, b, c, d = st.columns(4)
a.metric("Total sales", f'₹{data["sales"].sum():,.0f}')
b.metric("Total profit", f'₹{data["profit"].sum():,.0f}')
c.metric("Orders", f'{len(data):,}')
c_margin = (data["profit"].sum()/data["sales"].sum()*100) if data["sales"].sum() else 0
d.metric("Profit margin", f"{c_margin:.1f}%")

monthly = data.groupby("month")[["sales", "profit"]].sum().reset_index()
st.subheader("Monthly sales and profit")
fig, ax = plt.subplots(figsize=(11, 4))
sns.lineplot(data=monthly, x="month", y="sales", marker="o", label="Sales", ax=ax)
sns.lineplot(data=monthly, x="month", y="profit", marker="o", label="Profit", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Amount (₹)")
ax.tick_params(axis="x", rotation=35)
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.subheader("Product performance")
    product = data.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=product, x="sales", y="product", ax=ax)
    ax.set_xlabel("Sales (₹)")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Regional sales")
    regional = data.groupby("region", as_index=False)["sales"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pie(regional["sales"], labels=regional["region"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Share of sales by region")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Customer segmentation")
segment = data.groupby("customer_segment", as_index=False).agg(sales=("sales","sum"), orders=("order_id","nunique"))
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=segment, x="customer_segment", y="sales", ax=ax)
ax.set_xlabel("Customer segment")
ax.set_ylabel("Sales (₹)")
st.pyplot(fig)
plt.close(fig)

st.subheader("Summary table")
st.dataframe(data, use_container_width=True)
st.subheader("Findings")
best_product = product.iloc[0]
best_region = regional.sort_values("sales", ascending=False).iloc[0]
best_segment = segment.sort_values("sales", ascending=False).iloc[0]
st.write(f"- {best_product['product']} is the top product by sales in the selected data.")
st.write(f"- {best_region['region']} contributes the largest regional sales total.")
st.write(f"- {best_segment['customer_segment']} is the leading customer segment by sales.")
st.caption("Educational demo dataset. Replace it with your organization's verified sales records for actual business decisions.")
