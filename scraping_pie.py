import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from textblob import TextBlob

# ================== CONFIG ==================
INPUT_FILE = "komentar_video_final.csv"  # CSV komentar yang sudah ada
TOP_N_WORDS = 20
# ============================================

# ================== LOAD DATA ==================
df = pd.read_csv(INPUT_FILE)

# ================== ANALISIS SENTIMEN ==================
def label_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return "Positif"
    elif polarity < -0.05:
        return "Negatif"
    else:
        return "Netral"

df["Sentimen"] = df["Komentar"].apply(label_sentiment)

# Simpan CSV dengan kolom Sentimen
df.to_csv("komentar_dengan_sentimen.csv", index=False)
for s in ["Positif","Negatif","Netral"]:
    df[df["Sentimen"]==s].to_csv(f"komentar_{s.lower()}.csv", index=False)

# ================== PIE CHART DISTRIBUSI SENTIMEN ==================
counts = df["Sentimen"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["green","red","grey"])
plt.title("Distribusi Sentimen")
plt.savefig("pie_sentimen.png")
plt.close()
print("Pie chart distribusi sentimen tersimpan: pie_sentimen.png")

# ================== BAR CHART FREKUENSI KATA ==================
def plot_word_frequency(text, title, filename, top_n=TOP_N_WORDS):
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    counts = Counter(words)
    
    # Stopwords bahasa Indonesia + Inggris lebih lengkap
    stopwords = set([
        # Indonesia
        'yang','dan','di','atau','ini','itu','ke','dari','untuk','saya','kami','anda',
        'dengan','pada','adalah','karena','oleh','sebagai','sudah','belum','juga','tetapi',
        'agar','dalam','dari','pada','lagi','saat','atau','saja','ya','tidak','ya','lagi','yg','di',
        # Inggris
        'the','and','a','to','i','is','it','in','of','for','this','on','that',
        'you','with','my','me','was','are','but','so','at','be','as','have','they','do'
    ])
    
    for sw in stopwords:
        counts.pop(sw, None)

    most_common = counts.most_common(top_n)
    if not most_common:
        print(f"Tidak ada kata untuk chart: {title}")
        return

    kata, freq = zip(*most_common)
    plt.figure(figsize=(10,6))
    plt.barh(kata, freq, color="skyblue")
    plt.xlabel("Frekuensi")
    plt.title(title)
    plt.gca().invert_yaxis()  # kata terbanyak di atas
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Bar chart tersimpan: {filename}")


# Bar chart semua komentar
plot_word_frequency(" ".join(df["Komentar"]), "Top Kata di Semua Komentar", "bar_all.png")

# Bar chart per sentimen
for s in ["Positif","Negatif","Netral"]:
    text = " ".join(df[df["Sentimen"]==s]["Komentar"])
    if text.strip():
        plot_word_frequency(text, f"Top Kata Komentar {s}", f"bar_{s.lower()}.png")
