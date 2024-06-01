from pymongo import MongoClient

# Hubungkan ke MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Pilih database
db = client['Attendance']  # Ganti dengan nama database Anda

# Pilih koleksi
collection = db['Workers']  # Ganti dengan nama koleksi Anda

# Ambil dan cetak semua dokumen dalam koleksi
documents = collection.find()

for doc in documents:
    print(doc)
