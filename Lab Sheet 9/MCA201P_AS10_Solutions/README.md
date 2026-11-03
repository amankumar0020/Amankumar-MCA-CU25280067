# MCA201P-AS10 — Data Analytics & Visualization Mini Projects

This package implements **both projects** listed in the supplied Lab Sheet:
1. COVID-19 Data Analysis Dashboard
2. Sales Data Analysis Dashboard

## Requirements from the lab sheet covered
- Exploratory Data Analysis (EDA)
- Missing-value handling and duplicate removal
- Multiple charts with suitable chart types
- Country-wise / monthly / product / customer / regional analysis
- Written findings and explanations
- Source code and datasets included

## Setup
Use Python 3.9 or newer.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Run Project 1 — COVID-19
```bash
cd covid_dashboard
streamlit run app.py
```
Open the local URL shown by Streamlit. The sidebar filters countries.

## Run Project 2 — Sales
```bash
cd sales_dashboard
streamlit run app.py
```
Use the sidebar to filter region and customer segment.

## Dataset note
The included CSV files are **synthetic demonstration datasets** created for testing and showing the dashboard workflow. They are not official COVID-19 statistics or actual company sales. For a real submission requiring factual analysis, replace them with a verified dataset and rerun the dashboards.

## Submission checklist
- [x] Source code
- [x] Dataset files (demo data)
- [x] Dashboard screenshot examples
- [x] EDA and missing-value handling
- [x] Multiple charts
- [x] Findings and visualization explanations

## Visualization explanations
### COVID-19 dashboard
- Line chart: compares cumulative confirmed, recovered and death totals over time.
- Bar chart: compares latest confirmed-case totals by country.
- Vaccination line chart: shows the change in vaccination totals over time.
- KPI cards: provide a quick summary of the latest country totals.

### Sales dashboard
- Line chart: compares monthly sales and profit.
- Horizontal bar chart: compares sales across products.
- Pie chart: shows each region's share of sales.
- Segment bar chart: compares customer groups by sales.
- KPI cards: summarize sales, profit, order count and profit margin.
