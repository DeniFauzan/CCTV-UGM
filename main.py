import re
import whisper
from gtts import gTTS
from transformers import pipeline

# 1. Buat 3 Sampel Audio Uji Otomatis
sampel_kalimat = {
    "uji_normal.mp3": "Halo teman-teman, nanti kita kumpul di perpus pusat jam dua siang ya.",
    "uji_medium.mp3": "Anjir, susah banget soal ujian kalkulus hari ini, pusing kepala gue.",
    "uji_tinggi.mp3": "Dasar bodoh tidak berguna, awas saja kalau berani ke sini lagi!"
}

print("1. Membuat file audio uji...")
for nama_file, teks in sampel_kalimat.items():
    gTTS(text=teks, lang="id").save(nama_file)

# 2. Kamus Kata Kasar Lokal (Indonesia & Jawa/Jogja)
KAMUS_KATA_KASAR = {
    "anjing", "anjir", "babi", "bangsat", "goblok", "tolol", 
    "brengsek", "asu", "jancok", "cok", "bajingan", "bodoh"
}

# 3. Muat Model OpenAI Whisper
print("2. Memuat model Speech-to-Text (OpenAI Whisper)...")
stt_model = whisper.load_model("base")

# 4. Muat Model Deteksi Hate Speech / Toxicity
print("3. Memuat model Hate Speech / Toxicity Recognition...")
hs_classifier = pipeline(
    "text-classification", 
    model="unitary/multilingual-toxic-xlm-roberta"
)

# 5. Fungsi Klasifikasi Utama (Normal / Medium / Tinggi)
def analisis_audio_cctv(file_audio):
    # Tahap A: Speech-to-Text
    hasil_stt = stt_model.transcribe(file_audio, language="id")
    transkrip = hasil_stt["text"].strip()

    # Bersihkan tanda baca untuk pencocokan kamus
    kata_bersih = re.sub(r'[^\w\s]', '', transkrip.lower()).split()
    kata_kasar_ditemukan = [k for k in kata_bersih if k in KAMUS_KATA_KASAR]

    # Tahap B: Deteksi Hate Speech / Toxic dengan AI
    hasil_ai = hs_classifier(transkrip)[0]
    label_ai = hasil_ai['label']
    skor_ai = hasil_ai['score']

    # Tahap C: Logika Penentuan Tingkat Indikasi
    is_toxic = label_ai.lower() == "toxic"

    if is_toxic and skor_ai >= 0.80:
        status = "Tinggi"
    elif (is_toxic and skor_ai >= 0.50) or len(kata_kasar_ditemukan) > 0:
        status = "Medium"
    else:
        status = "Normal"

    return {
        "file": file_audio,
        "transkrip": transkrip,
        "kata_kasar": kata_kasar_ditemukan,
        "prediksi_ai": f"{label_ai} ({skor_ai:.2f})",
        "indikasi_status": status
    }

# 6. Jalankan Pengujian
print("\n--- HASIL ANALISIS CCTV AUDIO ---")
for file_uji in sampel_kalimat.keys():
    hasil = analisis_audio_cctv(file_uji)
    print(f"File       : {hasil['file']}")
    print(f"Transkrip  : {hasil['transkrip']}")
    print(f"Kata Kasar : {hasil['kata_kasar']}")
    print(f"Skor Model : {hasil['prediksi_ai']}")
    print(f"STATUS     : [{hasil['indikasi_status'].upper()}]")
    print("-" * 40)