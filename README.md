# RetailIQ - E-commerce Sales & Customer Analytics Platform

RetailIQ is an end-to-end e-commerce analytics project built around the Brazilian Olist e-commerce dataset.

The project combines data ingestion, data validation, PostgreSQL analytics, RFM customer segmentation, customer clustering, business recommendations, and Power BI visualization to analyze sales performance and customer behavior.

---

## Project Objectives

- Build a structured PostgreSQL database for e-commerce data
- Inspect and validate data before analysis
- Validate keys and relationships between datasets
- Develop reusable SQL business analytics
- Calculate customer RFM metrics
- Segment customers using K-Means clustering
- Validate customer clusters using clustering evaluation metrics
- Generate business recommendations from customer segments
- Build an interactive Power BI dashboard for business analysis

---

## Architecture

Olist E-commerce Dataset
          |
          v
     Raw CSV Data
          |
          v
   Python ETL & Validation
          |
          v
      PostgreSQL
       /       \
      v         v
SQL Analytics  RFM Analysis
                  |
                  v
           Customer Clustering
                  |
                  v
           Business Segments
                  |
                  v
              Power BI

---

## Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL |
| Query Language | SQL |
| Machine Learning | Scikit-learn |
| Clustering | K-Means |
| Visualization | Power BI, Matplotlib |
| Version Control | Git / GitHub |

---

## Project Structure

RetailIQ/
|
|-- data/
|   |-- raw/                         # Raw datasets, excluded from Git
|
|-- etl/
|   |-- inspect_dataset.py
|   |-- inspect_order_anomalies.py
|   |-- inspect_reviews.py
|   |-- load_to_postgres.py
|   |-- test_db_connection.py
|   |-- validate_keys.py
|   |-- validate_orders.py
|   |-- validate_relationships.py
|
|-- ml/
|   |-- rfm_clustering.py
|   |-- rfm_business_recommendations.py
|   |-- rfm_visualization.py
|
|-- sql/
|   |-- 01_schema.sql
|   |-- 02_analytics.sql
|
|-- backend/
|-- frontend/
|-- notebooks/
|-- tests/
|-- outputs/                         # Generated outputs, excluded from Git
|
|-- RetailIQ_RFM_Dashboard.pbix
|-- .gitignore
|-- README.md

---

## Data Pipeline

### 1. Data Inspection

The ETL layer contains Python scripts for inspecting the dataset, examining order anomalies, and inspecting review data before loading the data into PostgreSQL.

### 2. Data Validation

The project includes validation scripts for:

- Primary keys
- Foreign-key relationships
- Composite keys
- Order consistency
- Dataset relationships
- Database connectivity
- Review identifiers
- Customer identity relationships

### 3. PostgreSQL Database

The SQL schema creates a relational database structure for the e-commerce data.

Major entities include:

- Customers
- Orders
- Order Items
- Products
- Sellers
- Payments
- Reviews
- Geolocation

The schema uses primary keys, foreign keys, indexes, and validation constraints to maintain data integrity.

---

## RFM Customer Segmentation

RetailIQ uses RFM analysis based on three customer-level metrics.

### Recency

Measures how recently a customer purchased.

### Frequency

Measures how frequently a customer purchased.

### Monetary

Measures how much a customer spent.

The RFM clustering workflow:

1. Loads customer RFM data from PostgreSQL
2. Validates the input data
3. Examines RFM distributions and skewness
4. Caps extreme frequency values
5. Applies transformations to skewed variables
6. Standardizes the clustering features
7. Evaluates different cluster counts
8. Validates cluster stability
9. Produces the final customer segmentation
10. Generates outputs for downstream business recommendations and reporting

### Clustering

The project uses K-Means clustering for customer segmentation.

Clustering quality is evaluated using:

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

The final workflow produces four business customer clusters for downstream analysis.

---

## Business Recommendations

The RFM segmentation is used as the foundation for customer-targeting recommendations.

The project separates the analytical segmentation stage from the business recommendation stage so that customer segments can be interpreted and used for downstream business actions.

---

## Power BI Dashboard

The Power BI dashboard contains two analytical areas.

### RFM Customer Segmentation

The dashboard provides:

- Total Customers
- Total Revenue
- Average Revenue per Customer
- At Risk Customers
- Customer distribution by segment
- Revenue by segment
- Average revenue per customer by segment
- Average recency by segment
- Average frequency by segment
- RFM segment summary

### Sales & Product Performance

The dashboard provides:

- Total Sales
- Total Orders
- Average Order Value
- Total Items Sold
- Product Sales by Category
- Top 10 Product Categories by Sales
- Orders by Status
- Revenue by Year

---

## Key Business Questions

RetailIQ is designed to answer questions such as:

- How much revenue is being generated?
- How many orders and items are being sold?
- Which product categories generate the most sales?
- How does revenue change over time?
- How is the customer base distributed across RFM segments?
- Which customer segments generate more revenue?
- How do customer recency and purchase frequency differ between segments?
- Which customer groups can be targeted for different business actions?

---

## Data Security

Sensitive database credentials are stored through environment variables.

The .env file is excluded from version control using .gitignore.

Raw datasets and generated outputs are also excluded from Git where appropriate.

---

## Setup

### 1. Clone the Repository

git clone https://github.com/yuvi2207/RetailIQ.git
cd RetailIQ

### 2. Create a Python Environment

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

### 3. Install Required Python Packages

The project uses Python libraries including:

- pandas
- numpy
- psycopg2
- python-dotenv
- scikit-learn
- joblib
- matplotlib

Install them with:

pip install pandas numpy psycopg2-binary python-dotenv scikit-learn joblib matplotlib

### 4. Configure Environment Variables

Create a local .env file in the project root:

POSTGRES_HOST=
POSTGRES_PORT=
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=

Do not commit the .env file to GitHub.

### 5. Set Up PostgreSQL

Create the PostgreSQL database and execute:

sql/01_schema.sql

Then execute:

sql/02_analytics.sql

### 6. Run ETL and Validation

The scripts inside etl/ can be used to:

- Inspect the dataset
- Inspect order anomalies
- Inspect reviews
- Validate primary and foreign keys
- Validate relationships
- Validate orders
- Test the PostgreSQL connection
- Load data into PostgreSQL

### 7. Run RFM Analysis

The scripts inside ml/ perform:

- RFM clustering
- Cluster validation
- RFM visualization
- Business recommendations

### 8. Open the Power BI Dashboard

Open:

RetailIQ_RFM_Dashboard.pbix

The dashboard provides the final business-facing analysis of sales performance and customer segmentation.

---

## Project Outputs

The project produces analytical outputs related to:

- Data validation
- SQL business analytics
- RFM customer segmentation
- Cluster validation
- Customer visualization
- Business recommendations
- Power BI reporting

Generated outputs are excluded from Git using .gitignore.

---

## Future Improvements

Potential future improvements include:

- Automated ETL scheduling
- Automated data-quality testing
- Incremental data loading
- Automated Power BI refresh
- Customer lifetime value analysis
- Predictive customer churn modeling
- More advanced customer behavior modeling
- Deployment of an analytics API
- Production database and dashboard deployment

---

## Project Status

Completed

The current repository contains:

- Python ETL and validation scripts
- PostgreSQL database schema
- SQL analytics
- RFM customer segmentation
- K-Means clustering
- Cluster validation
- Business recommendation workflow
- RFM visualization
- Power BI dashboard

---

## Author

Yuvraj Vinayak Kadam

GitHub: https://github.com/yuvi2207
