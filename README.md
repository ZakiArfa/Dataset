# Dataset
Proyek ini merupakan implementasi pemrosesan data hasil serangan DDoS dari tiga sumber berbeda, yang kemudian digabungkan menjadi satu DataFrame. Dataset gabungan digunakan untuk pelatihan model klasifikasi menggunakan algoritma Decision Tree. Proyek ini mencakup proses preprocessing data, pemisahan fitur dan label, pelatihan model, evaluasi akurasi, serta visualisasi model pohon keputusan dan confusion matrix.

Import Library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as lol
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
Library yang digunakan:

pandas dan numpy: manipulasi data
matplotlib dan seaborn: visualisasi data
scikit-learn: pembelajaran mesin dan evaluasi model

Ekstraksi dan Pembacaan Data CSV
```python
import zipfile

with zipfile.ZipFile('drive-download-20250505T203935Z-001.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

dataset = pd.read_csv("DDoS UDP Flood.csv")
dataset2 = pd.read_csv("DoS ICMP Flood.csv")
dataset3 = pd.read_csv("DoS UDP Flood.csv")
```
Tiga file CSV yang berisi data serangan:

DDoS UDP Flood
DoS ICMP Flood
DoS UDP Flood
File berada dalam format ZIP dan diekstrak sebelum dibaca.

Penggabungan DataFrame
```python
hasilgabung = pd.concat([dataset, dataset2, dataset3])
```
Data dari ketiga sumber digabung menjadi satu DataFrame untuk pemrosesan selanjutnya.

Pemisahan Fitur dan Label
```python
x = hasilgabung.iloc[:, 7:76]
y = hasilgabung.iloc[:, 83:84]
```
Fitur (X): Kolom ke-7 hingga ke-75
Label (Y): Kolom ke-83, diasumsikan berisi target klasifikasi (jenis serangan)

Pembagian Data Latih dan Uji
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Data dibagi menjadi:

80% data latih
20% data uji

Pelatihan Model Decision Tree
```python
zaki = DecisionTreeClassifier(criterion='entropy', splitter='random')
zaki.fit(x_train, y_train)
y_pred = zaki.predict(x_test)
```
Model Decision Tree dibuat dan dilatih dengan parameter:

criterion='entropy': mengukur informasi
splitter='random': pembagian node secara acak

Evaluasi Akurasi
```python
accuracy = accuracy_score(y_test, y_pred)
```
Menghitung akurasi model dengan membandingkan hasil prediksi dan data aktual.

Visualisasi Pohon Keputusan
```python
fig = plt.figure(figsize=(10, 7))
tree.plot_tree(zaki, feature_names=x.columns.values, class_names=np.array(['DDoS UDP Flood.csv', 'DDos ICMP Flood', 'DDoS UDP Flood']), filled=True)
plt.show()
```
Struktur pohon keputusan divisualisasikan, menunjukkan aturan klasifikasi berdasarkan fitur.

Confusion Matrix
```python
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
label = np.array(['DDoS UDP Flood.csv', 'DDos ICMP Flood', 'DDoS UDP Flood'])
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
Menampilkan matriks kebingungan dalam bentuk heatmap, yang menggambarkan kinerja klasifikasi model terhadap masing-masing kelas.

Script UTS DATA MINING.py ini menyusun pipeline analitik lengkap untuk klasifikasi serangan jaringan berbasis data DDoS. Tahapan dimulai dari pembacaan dan penggabungan data, pemrosesan fitur, pelatihan model klasifikasi Decision Tree, hingga visualisasi hasil evaluasi. Model mampu melakukan klasifikasi otomatis terhadap jenis serangan jaringan berdasarkan fitur-fitur statistik. Visualisasi pohon keputusan dan confusion matrix membantu dalam mengevaluasi serta memahami pola pengambilan keputusan model.
