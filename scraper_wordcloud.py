from googleapiclient.discovery import build
import pandas as pd
import time
import sys
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# ====================== KONFIGURASI UTAMA ============================
API_KEY = "AIzaSyAR0On0soVckGtjdZI8zQfUSsX9E6SGEQg"
VIDEO_ID = "wbK0V2ASJf8"
OUTPUT_CSV = "komentar_video_final.csv"
OUTPUT_POS_CSV = "komentar_positif.csv"
OUTPUT_NEG_CSV = "komentar_negatif.csv"
OUTPUT_WORDCLOUD = "wordcloud_youtube.png"

SORT_ORDER = "time"
SLEEP_BETWEEN_REQUESTS = 0.1
# =====================================================================

# Download stopword
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))

print("=== Mulai skrip ===")

# =====================================================================
# Step 1: KONEKSI API
# =====================================================================
try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
    print("✔ Berhasil terhubung API YouTube")
except Exception as e:
    print("Gagal membuat service YouTube:", e)
    sys.exit(1)

# =====================================================================
# Step 2: FUNGSI AMBIL KOMENTAR
# =====================================================================

def get_all_comments(video_id):
    all_comments = []
    next_page_token = None
    total_threads = 0

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order=SORT_ORDER
            ).execute()
        except Exception as e:
            print("Error:", e)
            break

        items = response.get("items", [])
        if not items:
            break

        for item in items:
            # Komentar utama
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            all_comments.append({
                "Username": top_comment.get("authorDisplayName", ""),
                "Komentar": top_comment.get("textDisplay", ""),
                "Waktu": top_comment.get("publishedAt", ""),
                "IsReply": False
            })

            # Reply
            replies = item.get("replies", {}).get("comments", [])
            for reply in replies:
                rep = reply["snippet"]
                all_comments.append({
                    "Username": rep.get("authorDisplayName", ""),
                    "Komentar": rep.get("textDisplay", ""),
                    "Waktu": rep.get("publishedAt", ""),
                    "IsReply": True
                })

        total_threads += len(items)
        print(f"Progress: {total_threads} komentar...")

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_comments


print("Mengambil semua komentar...")
comments = get_all_comments(VIDEO_ID)

if not comments:
    print("Tidak ada komentar.")
    sys.exit(1)

# =====================================================================
# Step 3: SIMPAN CSV UTAMA
# =====================================================================
df = pd.DataFrame(comments)
df["Waktu"] = pd.to_datetime(df["Waktu"], errors="coerce")
df = df.sort_values(by="Waktu", ascending=False)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✔ CSV utama disimpan: {OUTPUT_CSV}")

# =====================================================================
# Step 4: ANALISIS SENTIMEN
# =====================================================================

def label_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positif"
    elif polarity < 0:
        return "Negatif"
    return "Netral"

df["Sentimen"] = df["Komentar"].apply(label_sentiment)

df_pos = df[df["Sentimen"] == "Positif"]
df_neg = df[df["Sentimen"] == "Negatif"]
df_net = df[df["Sentimen"] == "Netral"]

df_pos.to_csv(OUTPUT_POS_CSV, index=False, encoding="utf-8-sig")
df_neg.to_csv(OUTPUT_NEG_CSV, index=False, encoding="utf-8-sig")

print(f"✔ Positif: {len(df_pos)}")
print(f"✔ Negatif: {len(df_neg)}")
print(f"✔ Netral  : {len(df_net)}")

# =====================================================================
# Step 5: WORDCLOUD TOTAL + PER SENTIMEN
# =====================================================================

def clean_text(teks):
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\S+", "", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = " ".join([w for w in teks.split() if w not in stop_words])
    return teks

df["clean"] = df["Komentar"].apply(clean_text)

def buat_wordcloud(text, title):
    wc = WordCloud(width=1400, height=700, background_color="white").generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.show()

# WordCloud keseluruhan
buat_wordcloud(" ".join(df["clean"]), "WordCloud Semua Komentar")

# WordCloud per sentimen
buat_wordcloud(" ".join(df_pos["clean"]), "WordCloud Komentar Positif")
buat_wordcloud(" ".join(df_neg["clean"]), "WordCloud Komentar Negatif")
buat_wordcloud(" ".join(df_net["clean"]), "WordCloud Komentar Netral")

# =====================================================================
# Step 6: PIE CHART SENTIMEN
# =====================================================================

plt.figure(figsize=(7, 7))
plt.pie(
    [len(df_pos), len(df_net), len(df_neg)],
    labels=["Positif", "Netral", "Negatif"],
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Distribusi Sentimen Komentar")
plt.show()

# =====================================================================
# Step 7: BAR CHART FREKUENSI 20 KATA TERBANYAK
# =====================================================================

all_words = " ".join(df["clean"]).split()
word_freq = Counter(all_words).most_common(20)

words, counts = zip(*word_freq)

plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("20 Kata Paling Sering Muncul")
plt.xlabel("Kata")
plt.ylabel("Frekuensi")
plt.show()

print("=== Skrip selesai! 🎉 ===")
