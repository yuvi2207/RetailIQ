import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

connection = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)

print("PostgreSQL connection successful!")

cursor = connection.cursor()
cursor.execute("SELECT current_database(), current_user;")

database, user = cursor.fetchone()

print(f"Database: {database}")
print(f"User: {user}")

cursor.close()
connection.close()