"""
COVID-19 Data Analysis Dashboard
Run: streamlit run app.py
Dataset: covid_data.csv (sample/demo data)
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Analysis", layout="wide")
st.title("COVID-19 Data Analysis Dashboard")
st.caption("Exploratory analysis of confirmed cases, recoveries, deaths, vaccination trends and country statistics.")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_data.csv", parse_dates=["date"])
    df["country"] = df["country"].astype(str)
    return df

df = load_data()
# EDA / data cleaning
df = df.drop_duplicates()
for col in ["confirmed", "recovered", "deaths", "vaccinated"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(0).clip(lower=0)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "country"])

countries = sorted(df["country"].unique())
selected = st.sidebar.multiselect("Select countries", countries, default=countries)
view = df[df["country"].isin(selected)].copy()
if view.empty:
    st.warning("Select at least one country.")
    st.stop()

latest = view.sort_values("date").groupby("country").tail(1)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest confirmed", f'{int(latest["confirmed"].sum()):,}')
c2.metric("Latest recovered", f'{int(latest["recovered"].sum()):,}')
c3.metric("Latest deaths", f'{int(latest["deaths"].sum()):,}')
c4.metric("Latest vaccinated", f'{int(latest["vaccinated"].sum()):,}')

st.subheader("Daily trends")
daily = view.groupby("date")[["confirmed", "recovered", "deaths", "vaccinated"]].sum().reset_index()
fig, ax = plt.subplots(figsize=(11, 4))
for col in ["confirmed", "recovered", "deaths"]:
    sns.lineplot(data=daily, x="date", y=col, label=col.title(), ax=ax)
ax.set_ylabel("Cumulative people")
ax.set_xlabel("Date")
ax.tick_params(axis="x", rotation=25)
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.subheader("Country-wise confirmed cases")
    country_latest = latest.sort_values("confirmed", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=country_latest, x="confirmed", y="country", ax=ax)
    ax.set_xlabel("Confirmed cases")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Vaccination trend")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=daily, x="date", y="vaccinated", ax=ax, color="green")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vaccinated people")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Latest country statistics")
st.dataframe(country_latest[["country", "confirmed", "recovered", "deaths", "vaccinated"]], use_container_width=True)
st.subheader("Findings")
top = country_latest.iloc[0]
st.write(f"- {top['country']} has the highest confirmed-case total among the selected countries in this sample.")
st.write("- The trend chart compares cumulative confirmed, recovered and death counts over time.")
st.write("- Vaccination totals are displayed separately because they represent a different measure.")
st.caption("Educational demo dataset. Replace it with a verified public dataset before making real-world claims.")
