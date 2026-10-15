MCA201P-AS08 — Lab Sheet 07
Mini Projects on Unsupervised Learning Applications

Included projects
-----------------
1. Customer Segmentation using K-Means Clustering
2. News Article Clustering System using TF-IDF and clustering

Files
-----
customer_segmentation.py
customer_segmentation_dataset.csv
news_article_clustering.py
news_articles_dataset.csv
requirements.txt

Run in VS Code
--------------
1. Extract the ZIP and open the folder in VS Code.
2. Open Terminal > New Terminal.
3. Install packages:
   pip install -r requirements.txt
4. Run:
   python customer_segmentation.py
   python news_article_clustering.py

The programs clean data, scale/encode features where relevant, estimate a
cluster count using silhouette scores, compare K-Means and hierarchical
clustering, display metrics, save plots, and save a model with Joblib.

Important
---------
The supplied datasets are small illustrative examples for learning and viva
practice, not production-quality data. Customer income/spending values are
sample values. News articles are short demonstration texts. For reliable
business or news insights, use a larger authentic dataset.
