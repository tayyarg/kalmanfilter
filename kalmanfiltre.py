import numpy as np
import matplotlib.pyplot as plt
from math import *


# gaussian cizdiren yardimci fonksiyon tanimi
def gaussianpdf(ortalama, varyans, x):
    katsayi = 1.0 / sqrt(2.0 * pi *varyans)
    ustel = exp(-0.5 * (x-ortalama) ** 2 / varyans)
    return katsayi * ustel

# meta degiskenleri ilklendirelim
T = 15 # toplam surus suresi
dt = .1 # ornekleme periyodu

# Bayesci olmayan konum kestirimini hareketli-ortalama ile hesapladigimizi varsayalim
# asagidaki fonksiyon 5 uzunlugunda bir `window` kullanarak girdi olarak gelen sinyalin hareketli ortalamasini alir
har_ort_uzunluk = 5
def smooth(x,window_len=har_ort_uzunluk):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=np.ones(window_len,'d')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

# katsayi matrislerini tanimlayalim (dogrusal dinamik sistem katsayi matrisleri)
A = np.array([[1, dt], [0, 1]])  # durum gecis matrisi - aracin beklenen konum ve hizlarini temsilen
B = np.array([dt**2/2, dt]).reshape(2,1) # giris kontrol matrisi - giriste kontrollu olarak verilen ivmenin beklenen etkisini temsilen
C = np.array([1, 0]).reshape(1, 2) # gozlem matrisi - tahmin edilen durum elimizdeyken beklenen gozlemleri (olabilirlik) temsilen

# ana degiskenleri tanimlayalim
u=1.5 # ivmenin buyuklugu
OP_x=np.array([0,0]).reshape(2,1) # konum ve hizi temsil eden durum vektoru ilklendirme
OP_x_kest = OP_x # aracin ilklendirme esnasindaki durum kestirimi
OP_ivme_gurultu_buyuklugu = 0.05; # surec gurultusu - ivmenin standart deviasyonu - [m/s^2]
gozlem_gurultu_buyuklugu = 15;  # olcum gurultusu - otopilotun sensor olcum hatalari - [m]
Ez = gozlem_gurultu_buyuklugu**2; # olcum hatasini kovaryans matrisine cevirelim
Ex = np.dot(OP_ivme_gurultu_buyuklugu**2,np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])) # surec gurultusunu kovaryans matrisine cevirelim 
P = Ex; # ilk arac konum varyansinin kestirimi (kovaryans matrisi)

# sonuc degiskenlerini ilklendirelim
OP_konum = [] # aracin gercek konum vektoru
OP_hiz = [] # aracin gercek hiz vektoru
OP_konum_gozlem = [] # otopilotun gozlemledigi konum vektoru

# dt adimlariyla 0 dan T ye kadar simulasyonu calistiralim
for  t in np.arange(0, T, dt):

  # her bir adim icin aracin gercek durumunu hesaplayalim
  OP_ivme_gurultusu = np.array([[OP_ivme_gurultu_buyuklugu * i for i in np.array([(dt*2/2)*np.random.randn() , dt*np.random.randn()]).reshape(2,1)]]).reshape(2,1)
  OP_x = np.dot(A, OP_x)  + np.dot(B, u) +  OP_ivme_gurultusu

  # otopilotun gozlemledigi (olctugu) gurultulu konum vektorunu olusturalim
  gozlem_gurultusu = gozlem_gurultu_buyuklugu * np.random.randn()
  OP_z = np.dot(C, OP_x) + gozlem_gurultusu

  # konum, hiz ve gozlemleri cizdirmek icin vektor seklinde saklayalim 
  OP_konum.append(float(OP_x[0]))
  OP_hiz.append(float(OP_x[1]))
  OP_konum_gozlem.append(float(OP_z[0]))

# aracin gercek ve otopilot tarafindan gozlemlenen konumlarini cizdirelim
plt.plot(np.arange(0, T, dt), OP_konum, color='red', label='gercek konum')
plt.plot(np.arange(0, T, dt), OP_konum_gozlem, color='black', label='gozlenen konum')

# Kalman filtresi yerine klasik istatistik uygulayip Hareketli-Ortalama alan otopilotun tahmin ettigi konum
plt.plot(np.arange(0, T, dt), smooth(np.array(OP_konum_gozlem)[:-(har_ort_uzunluk-1)]), color='green', label='Klasik istatistik tahmini')
plt.ylabel('Konum [m]')
plt.xlabel('Zaman [s]')

plt.legend()
plt.show()

# Kalman Filtresi

# kestirim degiskenlerini ilklendirelim
OP_konum_kest = [] #otopilot pozisyon kestirimi
OP_hiz_kest = [] # otopilot hiz kestirimi
OP_x=np.array([0,0]).reshape(2,1) # otopilot durum vektorunu yeniden ilklendir
P_kest = P
P_buyukluk_kest = []
durum_tahmin = []
varyans_tahmin = []

for z in OP_konum_gozlem:
  
  # tahmin adimi

  # yeni durum tahminimizi hesaplayalim
  OP_x_kest = np.dot(A, OP_x_kest)  + np.dot(B, u)
  durum_tahmin.append(OP_x_kest[0])

  # yeni kovaryansi tahminini hesaplayalim
  P = np.dot(np.dot(A,P), A.T) + Ex
  varyans_tahmin.append(P)

  # guncelleme adimi
  
  # Kalman kazancini hesaplayalim
  K = np.dot(np.dot(P, C.T), np.linalg.inv(Ez + np.dot(C, np.dot(P, C.T))))

  # durum kestirimini guncelleyelim
  z_tahmin= z - np.dot(C, OP_x_kest)
  OP_x_kest = OP_x_kest + np.dot(K, z_tahmin)

  # kovaryans kestirimini guncelleyelim
  I = np.eye(A.shape[1])
  P = np.dot(np.dot(I - np.dot(K, C), P), (I - np.dot(K, C)).T) + np.dot(np.dot(K, Ez), K.T)

  #  otopilotun konum, hiz ve kovaryans tahminlerini vektorel olarak saklayalim 
  OP_konum_kest.append(np.dot(C, OP_x_kest)[0])
  OP_hiz_kest.append(OP_x_kest[1])
  P_buyukluk_kest.append(P[0])
 
plt.plot(np.arange(0, T, dt), OP_konum, color='red', label='gercek konum')
plt.plot(np.arange(0, T, dt), OP_konum_gozlem, color='black', label='gozlenen konum')
plt.plot(np.arange(0, T, dt), OP_konum_kest, color='blue', label='Bayesci Kalman tahmini')
plt.ylabel('Konum [m]')
plt.xlabel('Zaman [s]')
plt.legend()
plt.show()

# konumun mumkun olan araligini tanimlayalim
x_axis = np.arange(OP_x_kest[0]-gozlem_gurultu_buyuklugu*1.5, OP_x_kest[0]+gozlem_gurultu_buyuklugu*1.5, dt) 

# Kalman durum tahmin dagilimini bul
mu1 = OP_x_kest[0]
sigma1 = P[0][0]

print("Ortalama karesel hata: ", sigma1)

# durum tahmin dagilimini hesaplayalim
g1 = []
for x in x_axis:
    g1.append(gaussianpdf(mu1, sigma1, x))

# durum tahmin dagilimini cizdir
y=np.dot(g1,1/np.max(g1))
plt.plot(x_axis, y, label='sonsal tahmin dağılımı')
print(np.mean(x_axis))
print(OP_konum[-1])

# gozlemi dagilimini bul
mu2 = OP_konum_gozlem[-1]
sigma2 = gozlem_gurultu_buyuklugu

# gozlem dagilimini hesaplayalim
g2 = []
for x in x_axis:
    g2.append(gaussianpdf(mu2, sigma2, x))

# gozlem dagilimini cizdir
y=np.dot(g2,1/np.max(g2))
plt.plot(x_axis, y, label='gözlem dağılımı')

# gercek pozisyonu cizdir
plt.axvline(OP_konum[-1], 0.05, 0.95, color='red', label='gercek konum')
plt.legend(loc='upper left')
plt.xlabel('Konum [m]')
plt.ylabel('Olasılık Yoğunluk Fonksiyonu')
plt.show()