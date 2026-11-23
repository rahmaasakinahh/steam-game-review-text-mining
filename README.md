# **Steam Game Review Text Mining**

---

### **Analisis Ulasan Game Steam Menggunakan Natural Language Processing dan Apache Spark**

---

## **Daftar Isi**

---

1. [Gambaran Umum](#1-gambaran-umum)
2. [Tujuan Analisis](#2-tujuan-analisis)
3. [Dataset](#3-dataset)
4. [Teknologi yang Digunakan](#4-teknologi-yang-digunakan)
5. [Alur Analisis](#5-alur-analisis)
6. [Hasil Analisis](#6-hasil-analisis)
7. [Struktur Repository](#7-struktur-repository)
8. [Cara Menjalankan](#8-cara-menjalankan)
9. [Catatan Teknis dan Keterbatasan](#9-catatan-teknis-dan-keterbatasan)
10. [Referensi](#10-referensi)
11. [Kontributor](#11-kontributor)

## **1. Gambaran Umum**

---

**Steam Game Review Text Mining** merupakan proyek analisis teks ulasan pengguna game Steam menggunakan pendekatan **Natural Language Processing (NLP)** dan pemrosesan data dengan **Apache Spark (PySpark)**.

Proyek ini bertujuan mengeksplorasi karakteristik ulasan, mengolah teks menjadi data yang dapat dianalisis, mengidentifikasi **signature keywords** atau kata kunci yang menonjol untuk setiap game, serta mengelompokkan sentimen ulasan menjadi **positif, netral, dan negatif**.

Proses analisis dibagi menjadi dua Jupyter Notebook: bagian pertama berfokus pada **Exploratory Data Analysis (EDA)** dan *text preprocessing*, sedangkan bagian kedua membahas **pemrosesan teks dengan Spark, pembobotan TF-IDF, ekstraksi kata kunci, dan analisis sentimen**.

## **2. Tujuan Analisis**

---

1. Memahami struktur, kualitas, dan distribusi data ulasan game Steam.
2. Membersihkan serta menormalisasi teks agar siap diproses lebih lanjut.
3. Mengimplementasikan tokenisasi, *bigram*, dan pembobotan **Term Frequency–Inverse Document Frequency (TF-IDF)** menggunakan PySpark.
4. Mengidentifikasi kata kunci yang paling menonjol untuk masing-masing game berdasarkan akumulasi skor TF-IDF.
5. Mengklasifikasikan sentimen ulasan dan membandingkan distribusinya antar-game.
6. Menyajikan hasil eksplorasi dan analisis melalui visualisasi data.

## **3. Dataset**

---

Analisis menggunakan berkas `steam_game_reviews.csv` yang memuat ulasan pengguna dari berbagai game pada platform Steam.

### **3.1 Ringkasan Dataset**

---

| Informasi | Hasil |
|---|---:|
| Jumlah data awal | 992.153 ulasan |
| Jumlah kolom awal | 8 |
| Jumlah game unik | 242 |
| Data setelah penghapusan duplikat dan nilai kosong | 991.566 ulasan |
| Data setelah penyaringan ulasan berbahasa Inggris | 841.307 ulasan |

Nilai tersebut mengacu pada output yang tersimpan di notebook analisis.

### **3.2 Kolom Dataset Awal**

---

| Kolom | Keterangan |
|---|---|
| `review` | Teks ulasan pengguna. |
| `hours_played` | Informasi durasi bermain yang tercatat pada dataset. |
| `helpful` | Informasi penilaian kegunaan ulasan. |
| `funny` | Informasi penilaian humor pada ulasan. |
| `recommendation` | Informasi rekomendasi pengguna terhadap game. |
| `date` | Tanggal ulasan. |
| `game_name` | Nama game yang diulas. |
| `username` | Nama pengguna yang menulis ulasan. |

Analisis lanjutan berfokus pada kolom `game_name`, `username`, dan `review`, kemudian membentuk kolom teks hasil pembersihan `review_cleaned`.

**Catatan dataset:** Dataset mentah berukuran sekitar **453,75 MB** pada penyimpanan lokal. Berkas tersebut tidak disertakan dalam repository Git biasa karena melebihi batas ukuran file GitHub. Pengguna perlu menyediakan dataset secara terpisah untuk menjalankan ulang analisis. Sumber dan ketentuan penggunaan dataset perlu dicantumkan setelah tautan sumber aslinya dipastikan.

## **4. Teknologi yang Digunakan**

---

| Teknologi | Kegunaan |
|---|---|
| **Python** | Bahasa pemrograman utama untuk analisis data. |
| **Jupyter Notebook** | Lingkungan eksperimen dan dokumentasi alur analisis. |
| **Pandas & NumPy** | Manipulasi data tabular dan operasi numerik. |
| **Apache Spark (PySpark)** | Pemrosesan data, transformasi, dan agregasi ulasan. |
| **Spark MLlib** | Tokenisasi, *N-gram*, `CountVectorizer`, serta perhitungan IDF dan TF-IDF. |
| **NLTK** | *Part-of-speech tagging* dan lemmatisasi teks. |
| **langdetect** | Identifikasi bahasa ulasan. |
| **contractions** | Ekspansi kontraksi bahasa Inggris pada teks. |
| **TextBlob** | Analisis polaritas untuk klasifikasi sentimen. |
| **Matplotlib & Seaborn** | Pembuatan grafik eksplorasi dan hasil analisis. |
| **WordCloud** | Visualisasi frekuensi dan kemunculan kata. |

## **5. Alur Analisis**

---

### **5.1 Exploratory Data Analysis dan Data Cleaning**

---

Tahap awal dilakukan pada `Part1_5025241201_5025241222.ipynb` dengan langkah berikut:

1. Memuat dataset dan memeriksa jumlah baris, kolom, tipe data, serta statistik deskriptif.
2. Memilih tiga kolom utama: `game_name`, `username`, dan `review`.
3. Menghapus data duplikat dan baris yang tidak memiliki `review` atau `username`.
4. Mengeksplorasi distribusi jumlah ulasan per game dan membuat grafik **10 game dengan ulasan terbanyak**.
5. Memperluas kontraksi teks bahasa Inggris, mengubah teks menjadi huruf kecil, serta menghapus URL, angka, tanda baca, karakter tunggal, dan spasi berlebih.
6. Mendeteksi bahasa ulasan dan mempertahankan teks berbahasa Inggris.
7. Melakukan **lemmatisasi berbasis POS tagging** menggunakan NLTK.
8. Menyimpan hasil pemrosesan sebagai `df_selected_cleaned.csv` untuk digunakan pada notebook kedua.

### **5.2 Tokenisasi dan Pembentukan Fitur Teks**

---

Tahap lanjutan pada `Part2_5025241201_5025241222.ipynb` memuat hasil pembersihan ke **Spark DataFrame** dan menggunakan:

- **RegexTokenizer** untuk memisahkan teks ulasan menjadi token.
- **NGram** untuk membentuk pasangan kata (*bigram*).
- **CountVectorizer** dan **IDF** untuk merepresentasikan token serta menghitung skor TF-IDF.

Notebook juga mengeksplorasi pendekatan `HashingTF`, kemudian beralih ke **CountVectorizer** agar token dapat ditelusuri kembali melalui daftar kosakata (*vocabulary*).

### **5.3 Seleksi Kata dan Signature Keywords**

---

Pemilihan kata dilakukan dengan mempertimbangkan distribusi nilai IDF. Setelah melakukan analisis distribusi dan perbandingan ambang batas, notebook menggunakan **`IDF_THRESHOLD = 3.0`** sebagai nilai seleksi eksploratif.

Token yang memenuhi ambang batas dipasangkan dengan skor TF-IDF, lalu skor setiap kata diakumulasikan per game menggunakan operasi **Spark RDD** seperti `map`, `reduceByKey`, dan `groupByKey`.

Hasil akhirnya berupa **tiga signature keywords dengan total skor tertinggi** untuk masing-masing game.

### **5.4 Analisis Sentimen**

---

Sentimen teks dianalisis menggunakan **TextBlob** melalui skor polaritas, dengan aturan yang diterapkan dalam notebook:

| Skor Polaritas | Kategori |
|---|---|
| Lebih dari `0.1` | **Positive** |
| Kurang dari `-0.1` | **Negative** |
| Antara `-0.1` sampai `0.1` (inklusif) | **Neutral** |

Distribusi sentimen dihitung untuk setiap game dan divisualisasikan dalam grafik batang bertumpuk pada **15 game dengan ulasan terbanyak**.

## **6. Hasil Analisis**

---

### **6.1 Ringkasan Hasil**

---

| Tahap | Hasil |
|---|---|
| **Data Cleaning** | Data berkurang dari 992.153 menjadi 991.566 ulasan setelah penghapusan duplikat dan nilai kosong. |
| **Language Filtering** | Tersisa 841.307 ulasan berbahasa Inggris berdasarkan deteksi bahasa. |
| **Text Processing** | Teks diproses melalui normalisasi, lemmatisasi, tokenisasi, dan pembentukan bigram. |
| **TF-IDF** | Pembobotan kata dilakukan menggunakan fitur Spark MLlib dan pemilihan berdasarkan nilai IDF. |
| **Keyword Extraction** | Tiga signature keywords diekstraksi untuk setiap game; notebook mencatat hasil untuk 242 game. |
| **Sentiment Analysis** | Ulasan dikelompokkan menjadi positif, netral, dan negatif berdasarkan polaritas TextBlob. |

### **6.2 Contoh Signature Keywords**

---

Berikut contoh keluaran dari notebook:

| Game | Signature Keywords |
|---|---|
| **7 Days to Die** | `zombie`, `alpha`, `survival` |
| **American Truck Simulator** | `truck`, `drive`, `ets` |
| **Age of Empires II: Definitive Edition** | `aoe`, `age`, `rts` |
| **Assetto Corsa Competizione** | `race`, `gt`, `car` |

Kata kunci di atas merupakan hasil pembobotan dan agregasi pada dataset yang digunakan, bukan label kategori yang ditentukan secara manual.

### **6.3 Visualisasi**

---

Visualisasi yang tersedia di dalam notebook mencakup:

- Grafik 10 game dengan jumlah ulasan terbanyak.
- *Word cloud* ulasan setelah pembersihan teks.
- Histogram dan *boxplot* distribusi nilai IDF.
- Distribusi skor TF-IDF.
- Grafik jumlah *signature keywords* per game dan *word cloud* kata kunci.
- Grafik distribusi sentimen untuk 15 game dengan ulasan terbanyak.

## **7. Struktur Repository**

---

Struktur repository yang disarankan untuk publikasi:

```text
steam-game-review-text-mining/
├── Part1_5025241201_5025241222.ipynb
├── Part2_5025241201_5025241222.ipynb
├── .gitignore
└── README.md
```

| File | Deskripsi |
|---|---|
| `Part1_5025241201_5025241222.ipynb` | EDA, data cleaning, deteksi bahasa, dan lemmatisasi ulasan. |
| `Part2_5025241201_5025241222.ipynb` | Pemrosesan PySpark, TF-IDF, signature keywords, analisis sentimen, dan visualisasi. |
| `.gitignore` | Mengecualikan dataset CSV dan file sementara dari commit Git. |
| `README.md` | Dokumentasi proyek dan panduan menjalankan analisis. |

Berkas `steam_game_reviews.csv` dan `df_selected_cleaned.csv` digunakan secara lokal dan **tidak perlu diunggah** ke repository.

## **8. Cara Menjalankan**

---

### **8.1 Prasyarat**

---

Pastikan tersedia:

1. **Python** dan **Jupyter Notebook** (notebook pertama tercatat menggunakan Python 3.11).
2. **Java/JDK** yang kompatibel dengan versi PySpark yang digunakan.
3. **Git** untuk melakukan *clone* repository.
4. Ruang penyimpanan dan memori yang memadai untuk memproses dataset berukuran besar.
5. Berkas dataset `steam_game_reviews.csv` dari sumber yang sesuai.

Notebook pertama menggunakan **PySpark 3.5.1**. Kompatibilitas Python, Java, dan PySpark perlu diperiksa sesuai lingkungan lokal masing-masing.

### **8.2 Clone Repository**

---

Jalankan perintah berikut:

```bash
git clone https://github.com/USERNAME/steam-game-review-text-mining.git
cd steam-game-review-text-mining
```

Ganti `USERNAME` dengan username GitHub pemilik repository.

### **8.3 Instalasi Dependensi**

---

Instal pustaka yang dipakai dalam notebook:

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn pyspark==3.5.1 nltk langdetect contractions wordcloud textblob
```

Unduh sumber daya NLTK yang diperlukan:

```python
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")
```

### **8.4 Menyiapkan Dataset dan Path File**

---

Letakkan `steam_game_reviews.csv` pada direktori yang sama dengan kedua notebook. **Notebook asli masih menggunakan path absolut Windows**, sehingga referensi file perlu disesuaikan sebelum dijalankan ulang.

Pada **Part 1**, gunakan path relatif untuk membaca data:

```python
data = pd.read_csv("steam_game_reviews.csv")
```

Pada sel ekspor hasil preprocessing di **Part 1**, gunakan:

```python
df_selected.to_csv("df_selected_cleaned.csv", index=False)
```

Pada **Part 2**, gunakan:

```python
data = pd.read_csv("df_selected_cleaned.csv")
```

Contoh isi `.gitignore` untuk mencegah CSV besar ikut terunggah:

```gitignore
steam_game_reviews.csv
df_selected_cleaned.csv
.ipynb_checkpoints/
__pycache__/
```

### **8.5 Menjalankan Notebook**

---

Buka Jupyter Notebook:

```bash
jupyter notebook
```

Jalankan notebook secara berurutan:

1. **`Part1_5025241201_5025241222.ipynb`** — menghasilkan `df_selected_cleaned.csv`.
2. **`Part2_5025241201_5025241222.ipynb`** — membaca hasil Part 1 dan menjalankan analisis lanjutan.

Pastikan direktori kerja Jupyter sesuai dengan lokasi dataset dan notebook. Proses PySpark dapat memerlukan waktu dan memori cukup besar.

## **9. Catatan Teknis dan Keterbatasan**

---

- **Reproduksibilitas:** Notebook dalam arsip proyek masih memakai path dataset lokal. Path tersebut perlu disesuaikan seperti pada Bagian 8 sebelum analisis dapat dijalankan di komputer lain.
- **Ketersediaan dataset:** Berkas CSV mentah tidak disertakan pada repository Git biasa karena ukurannya melebihi batas unggah file. Sumber dataset perlu didokumentasikan secara terpisah.
- **Eksekusi Apache Spark:** Beberapa output yang tersimpan pada notebook memperlihatkan *warning* terkait ukuran task dan error koneksi/worker Spark. Sebagian hasil analisis berhasil dihasilkan, tetapi seluruh sel belum dapat dinyatakan berjalan tanpa kendala dari awal hingga akhir.
- **Pemilihan ambang IDF:** Nilai `3.0` merupakan keputusan eksploratif berdasarkan pemeriksaan distribusi pada dataset, bukan ambang universal untuk semua korpus teks.
- **Analisis sentimen:** Kategori sentimen berasal dari polaritas **TextBlob** berbasis aturan leksikal, bukan model klasifikasi yang dilatih dan dievaluasi menggunakan label sentimen manual.

## **10. Referensi**

---

1. [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
2. [PySpark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
3. [Pandas Documentation](https://pandas.pydata.org/docs/)
4. [NLTK Documentation](https://www.nltk.org/)
5. [TextBlob Documentation](https://textblob.readthedocs.io/)
6. [Matplotlib Documentation](https://matplotlib.org/stable/)
7. [Jupyter Documentation](https://docs.jupyter.org/)

## **11. Kontributor**

---

Proyek ini merupakan hasil kerja kolaboratif **Nasyita Larasati E.** dan **Rahma Sakinah**, sebagaimana tercantum pada notebook proyek.
