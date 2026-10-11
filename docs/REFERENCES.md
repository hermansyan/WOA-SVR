# Referensi

## 1. Whale Optimization Algorithm (rujukan utama metode optimasi)

**Mirjalili, S. & Lewis, A. (2016).** "The Whale Optimization Algorithm."
*Advances in Engineering Software*, 95, 51–67.
DOI: [10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008)

Paper asli yang memperkenalkan WOA. Tiga mekanisme yang diimplementasikan
di `woa.py` berasal langsung dari paper ini:

- *Encircling prey* (Eq. 2.1–2.2): `X(t+1) = X*(t) − A·D`, `D = |C·X*(t) − X(t)|`
- *Spiral bubble-net* (Eq. 2.5): `X(t+1) = D'·e^(bl)·cos(2πl) + X*(t)`
- *Search for prey* (Eq. 2.7–2.8): update relatif terhadap agen acak
- Koefisien (Eq. 2.3–2.4): `A = 2a·r − a`, `C = 2·r`, dengan `a` turun
  linear dari 2 ke 0 sepanjang iterasi
- Aturan pemilihan (Eq. 2.6): `p < 0.5` → shrinking/search, `p ≥ 0.5` → spiral

Kode sumber resmi penulis: http://www.alimirjalili.com/WOA.html

## 2. WOA untuk optimasi model machine learning (preseden metode)

**Li, L., Sun, J., Tseng, M. L. & Li, Z. (2019).**
"Extreme learning machine optimized by whale optimization algorithm using
insulated gate bipolar transistor module aging degree evaluation."
*Expert Systems with Applications*, 127, 58–67.
DOI: [10.1016/j.eswa.2019.03.002](https://doi.org/10.1016/j.eswa.2019.03.002)

Menunjukkan pola yang sama dengan repo ini — WOA dipakai sebagai
*hyperparameter/weight optimizer* untuk model prediktif — dan membuktikan
WOA-ELM mengungguli GA-ELM, CS-ELM, dan DA-ELM pada data eksperimen.

## 3. Riset terkait penulis (program riset yang sama)

**Syantoso, H. & Soebroto, A. A.**
"Peramalan Kasus Positif Harian Covid-19 di Indonesia Menggunakan Metode
Extreme Learning Machine (ELM) dengan Optimasi Whale Optimization Algorithm (WOA)."
*Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer (J-PTIIK),
Universitas Brawijaya.*

Skripsi/jurnal penulis memakai pasangan metode yang sama (WOA sebagai
optimizer) untuk peramalan kasus harian Covid-19 Indonesia
(13 Mar 2020 – 23 Jun 2021, data kawalcovid19.id). Konfigurasi optimal yang
dilaporkan: 6 fitur, 20 hidden neuron, 100 populasi paus, 50 iterasi —
MAPE 11,790% (WOA-ELM) vs 13,502% (ELM tanpa optimasi).

> Repo ini mengkaji varian **SVR** (bukan ELM) dengan optimizer WOA yang
> sama, sehingga kedua karya membentuk studi banding yang alami:
> ELM-WOA vs SVR-WOA pada domain peramalan deret waktu.

## 4. Referensi pendukung

- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2004/2006). Extreme Learning Machine
  — fondasi teori ELM yang dirujuk jurnal di atas.
- Alameer, Z. et al. (2019). Forecasting gold price fluctuations using improved
  MLP and WOA. *Resources Policy*, 61, 250–260. — contoh lain WOA untuk
  optimasi model prediktif.
