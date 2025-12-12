# ============================================================
# IMPORT LIBRARY
# ============================================================

from googleapiclient.discovery import build  
# → Library resmi Google untuk mengakses YouTube Data API (scraping komentar)

import pandas as pd  
# → Dipakai untuk menyimpan data dalam bentuk tabel (DataFrame) dan CSV

import time  
# → Memberi jeda antar request API agar tidak terkena limit

import sys  
# → Digunakan untuk menghentikan program jika error fatal

import re  
# → Regular expression untuk membersihkan teks komentar

import nltk  
# → Library NLP untuk bahasa alami (Natural Language Processing)

from nltk.corpus import stopwords  
# → Digunakan untuk mengambil daftar stopword bahasa Indonesia

from textblob import TextBlob  
# → Library untuk analisis sentimen (positif, negatif, netral)

# ============================================================
# DOWNLOAD STOPWORD (dijalankan 1x)
# ============================================================
nltk.download('stopwords')  
stop_words = set(stopwords.words('indonesian'))  
# → stop_words akan digunakan saat proses cleaning komentar


# ============================================================
# KONFIGURASI UTAMA
# ============================================================
API_KEY = "AIzaSyAR0On0soVckGtjdZI8zQfUSsX9E6SGEQg"   # API Key YouTube
VIDEO_ID = "wbK0V2ASJf8"                               # ID Video YouTube
OUTPUT_FILE = "komentar_video_final.csv"               # File output utama
SORT_ORDER = "time"                                    # Urutkan berdasarkan waktu
SLEEP_BETWEEN_REQUESTS = 0.1                           # Delay antar request

print("=== Mulai skrip ===")


# ============================================================
# STEP 1: MEMBUAT SERVICE YOUTUBE
# ============================================================
try:
    print("Membuat service YouTube...")
    youtube = build("youtube", "v3", developerKey=API_KEY)  
    # → Inisialisasi akses API
    print("Service YouTube berhasil dibuat!")
except Exception as e:
    print("Gagal membuat service YouTube:", e)
    sys.exit(1)



# ============================================================
# STEP 2: TEST AMBIL 1 KOMENTAR
# ============================================================
try:
    test_response = youtube.commentThreads().list(
        part="snippet",
        videoId=VIDEO_ID,
        maxResults=1,
        textFormat="plainText"
    ).execute()

    if test_response.get("items"):
        print("Berhasil mengambil komentar pertama!")
    else:
        print("Video tidak memiliki komentar atau komentar dimatikan.")
except Exception as e:
    print("Gagal mengambil komentar:", e)
    sys.exit(1)



# ============================================================
# STEP 3: AMBIL SEMUA KOMENTAR (FUNGSI)
# ============================================================
def get_all_comments(video_id):
    all_comments = []  
    next_page_token = None  

    while True:
        try:
            # Request 1 halaman komentar (max 100)
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order=SORT_ORDER
            ).execute()
        except Exception as e:
            print("Gagal mengambil komentar:", e)
            break

        items = response.get("items", [])
        if not items:
            break

        # Loop setiap komentar utama
        for item in items:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]

            all_comments.append({
                "Username": top_comment.get("authorDisplayName", ""),
                "Komentar": top_comment.get("textDisplay", ""),
                "Waktu": top_comment.get("publishedAt", ""),
                "IsReply": False
            })

            # Ambil balasan komentar (jika ada)
            replies = item.get("replies", {}).get("comments", [])
            for reply in replies:
                reply_snippet = reply["snippet"]
                all_comments.append({
                    "Username": reply_snippet.get("authorDisplayName", ""),
                    "Komentar": reply_snippet.get("textDisplay", ""),
                    "Waktu": reply_snippet.get("publishedAt", ""),
                    "IsReply": True
                })

        # Cek apakah ada halaman selanjutnya?
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_comments



print("\nMulai mengambil semua komentar...")
comments = get_all_comments(VIDEO_ID)

if not comments:
    print("Tidak ada komentar yang berhasil diambil.")
    sys.exit(1)



# ============================================================
# STEP 4: KONVERSI LIST → DATAFRAME
# ============================================================
df = pd.DataFrame(comments)

df["Waktu"] = pd.to_datetime(df["Waktu"])  
# → Convert waktu ke format datetime agar bisa diurutkan

df = df.sort_values(by="Waktu", ascending=False)



# ============================================================
# STEP 5: PEMBERSIHAN DATA (DATA CLEANING)
# ============================================================
print("\n=== Membersihkan data komentar... ===")

# Hapus duplikat
df = df.drop_duplicates(subset=["Username", "Komentar"])

# Filter SPAM berdasarkan kata kunci
spam_keywords = ["promo", "slot", "wa.me", "online", "dapet uang", "pinjaman"]
pattern_spam = "|".join(spam_keywords)
df = df[~df["Komentar"].str.lower().str.contains(pattern_spam, na=False)]



# ============================================================
# STEP 6: CLEANING TEKS KOMENTAR
# ============================================================
def clean_text(text):

    text = text.lower()  # Ubah ke huruf kecil semua

    text = re.sub(r"http\S+|www\S+", "", text)  # Hapus URL

    text = re.sub(r"[^\w\s]", " ", text)  # Hapus emoji, simbol, tanda baca

    text = re.sub(r"\d+", " ", text)  # Hapus angka

    words = text.split()

    # Hapus stopword (dan, yang, di, ke, dll)
    words = [w for w in words if w not in stop_words]

    return " ".join(words).strip()


df["Cleaned"] = df["Komentar"].apply(clean_text)

df.to_csv("komentar_bersih.csv", index=False)
print(f"Data bersih disimpan: komentar_bersih.csv (Total: {len(df)})")



# ============================================================
# STEP 7: ANALISIS SENTIMEN
# ============================================================
def label_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity  
    # → Polarity: -1 (negatif) sampai 1 (positif)

    if polarity > 0:
        return "Positif"
    elif polarity < 0:
        return "Negatif"
    else:
        return "Netral"


df["Sentimen"] = df["Cleaned"].apply(label_sentiment)

# Pisahkan berdasarkan kategori
df_pos = df[df["Sentimen"] == "Positif"]
df_neg = df[df["Sentimen"] == "Negatif"]

# Simpan
df_pos.to_csv("komentar_positif.csv", index=False)
df_neg.to_csv("komentar_negatif.csv", index=False)

print(f"\nTotal komentar bersih: {len(df)}")
print(f"Total positif: {len(df_pos)}")
print(f"Total negatif: {len(df_neg)}")

print("\n=== Skrip selesai ===")
