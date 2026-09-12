# Data source

Daqing Chen (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.

- Source: https://doi.org/10.24432/C5BW33
- License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Download the workbook separately. To use the original loading code, name it `Online_Retail.xlsx`, place it in `notebooks/`, and launch Jupyter from that folder. Do not upload the workbook to the repository; `.gitignore` excludes it.

I built a customer segmentation project in Python using online retail transaction data. I cleaned the dataset, removed cancelled and invalid transactions, calculated Revenue, and created Recency, Frequency, and Monetary (RFM) features for each customer. I scaled and transformed the data, then used K-Means clustering to group customers with similar purchasing behaviour.
