from src.data_ingestion import DataIngestion
data = DataIngestion()
df = data.load_data()
print(df.shape)