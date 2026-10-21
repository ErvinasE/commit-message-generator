# 1.4 užduotis: kaip veikia `cnn.ipynb`

Šiame aplanke:

| Failas | Kas tai |
|---|---|
| `EXPLANATION.md` | šis paaiškinimas: kaip veikia `cnn.ipynb`, langelis po langelio |
| `cnn_FashionMNIST.ipynb` | perdarytas sąsiuvinis su Fashion-MNIST prekių nuotraukomis |
| `data/fashion-mnist_train.csv.gz`, `data/fashion-mnist_test.csv.gz` | 6 000 + 1 000 Fashion-MNIST nuotraukų CSV formatu (Kaggle stulpeliai) |

Langelių numeriai `[n]` skaičiuojami nuo 0 ir apima ir teksto, ir kodo langelius.

---

## Bendra idėja

Sąsiuvinis moko **konvoliucinį neuroninį tinklą (CNN)** atpažinti CIFAR-10 nuotraukas ir parodo, kodėl konvoliucija
tinka vaizdams geriau nei paprastas tinklas. Jis turi tris dalis:

1. **1 dalis**: CNN mokymas ir įvertinimas (~70 % tikslumas);
2. **2 dalis**: ką „mato“ tinklas: filtrų ir požymių žemėlapių vizualizacija, o konvoliucija perskaičiuojama ranka su `numpy`;
3. **3 dalis**: ta pati užduotis su daugiasluoksniu perceptronu (MLP) be konvoliucijų (~43 % tikslumas).

**Duomenys: CIFAR-10.** 60 000 spalvotų 32×32 taškų nuotraukų, 10 klasių (lėktuvas, automobilis, paukštis, katė,
elnias, šuo, varlė, arklys, laivas, sunkvežimis), po 6 000 kiekvienos. 50 000 mokymui, 10 000 testavimui.

---

## 1 DALIS. CNN mokymas

### Duomenys

* **[0]–[3]** Logotipas, autorius, CIFAR-10 antraštė.
* **[4]** Importai (`tensorflow`, `keras.datasets/layers/models`, `matplotlib`, `numpy`) ir 10 klasių pavadinimų sąrašas
  `class_names`. Standartinis įkėlimas `datasets.cifar10.load_data()` užkomentuotas: komentare nurodyta, kad atsisiuntimas
  per Keras užtrunka 40+ min.
* **[6]** Vietoje to CIFAR-10 įkeliamas iš **Hugging Face** (`datasets.load_dataset("uoft-cs/cifar10")`).
  Kiekviena PIL nuotrauka paverčiama `numpy` masyvu ir visos sudedamos į vieną masyvą (`np.stack`):
  `train_images` forma `(50000, 32, 32, 3)`, `test_images` `(10000, 32, 32, 3)`. Žymės paverčiamos `(N, 1)` stulpeliu,
  kaip grąžintų Keras.
* **[7]** **Normalizavimas:** taškų reikšmės 0–255 padalijamos iš 255, gaunamos reikšmės 0–1. Tinklas greičiau ir stabiliau
  mokosi su mažais skaičiais.
* **[9]** Nupiešiamos pirmosios 100 mokymo nuotraukų (10×10 tinklelis) su klasių pavadinimais, kad įsitikintume,
  jog nuotraukos ir žymės atitinka.

### Modelis

* **[12]** `Sequential` CNN:

  | Sluoksnis | Išėjimo forma | Parametrai | Ką daro |
  |---|---|---|---|
  | `Conv2D(32, (3,3), relu)` | 30×30×32 | 896 | 32 filtrai 3×3×3; be užpildymo 32 → 30 |
  | `MaxPooling2D((2,2))` | 15×15×32 | 0 | iš kiekvieno 2×2 kvadrato palieka didžiausią reikšmę |
  | `Conv2D(64, (3,3), relu)` | 13×13×64 | 18 496 | sudėtingesni raštai iš ankstesnių |
  | `MaxPooling2D((2,2))` | 6×6×64 | 0 | |
  | `Conv2D(64, (3,3), relu)` | 4×4×64 | 36 928 | |
  | `Flatten` | 1 024 | 0 | ištiesina į vektorių |
  | `Dense(64, relu)` | 64 | 65 600 | klasifikatorius |
  | `Dense(10)` | 10 | 650 | 10 klasių logitai |

  **Konvoliucija:** filtras (mažas svorių masyvas, čia 3×3×3) slenka per nuotrauką; kiekvienoje vietoje apskaičiuojama
  filtro ir nuotraukos gabalėlio sandaugų suma + poslinkis, tada ReLU. Rezultatas yra **požymių žemėlapis**, rodantis,
  kur nuotraukoje yra filtro ieškomas raštas. Tas pats filtras naudojamas visose vietose, todėl parametrų mažai,
  o raštas atpažįstamas bet kurioje nuotraukos vietoje.
  Parametrų formulė: (3·3·įėjimo kanalai + 1)·filtrų skaičius, pvz. (3·3·3 + 1)·32 = 896.

  Užkomentuota alternatyva (`Flatten` + `Dense(10)`) yra paprasčiausias tiesinis klasifikatorius.
  Įspėjimas apie `input_shape` reiškia, kad Keras 3 rekomenduoja pirmuoju sluoksniu dėti `Input(shape)`.
* **[13]** `compile`: optimizatorius `adam`, nuostolis `SparseCategoricalCrossentropy(from_logits=True)` (daug klasių,
  žymė yra klasės numeris, išėjimas yra logitai), metrika `accuracy`. `summary()`: **122 570 parametrų**.
* **[14]** `plot_model` nupiešia sluoksnių schemą su formomis.

### Mokymas ir įvertinimas

* **[16]** `model.fit(..., epochs=20, validation_data=(test_images, test_labels))`: 20 epochų, kiekvienoje
  1 563 paketai po 32 nuotraukas. **Pastaba:** validavimui naudojama **testavimo** aibė, todėl galutinis testas
  nebėra visiškai „nematytais“ duomenimis (modelis pagal jį nederinamas, bet metodiškai teisingiau turėti atskirą validavimo aibę).
* **[18]** Mokymo ir validavimo tikslumo kreivės + `evaluate`: **tikslumas 0,6976 (~70 %)**. Mokymo išvestyje matyti
  persimokymas: validavimo nuostolis 6 epochoje buvo 0,876, o pabaigoje 1,143, nors mokymo tikslumas vis dar kilo.

---

## 2 DALIS. Filtrų ir požymių žemėlapių vizualizavimas

* **[21]** Paaiškinimas: filtrai yra svoriai, juos galima nupiešti kaip paveikslėlius; filtro gylis turi sutapti su įėjimo kanalų skaičiumi.
* **[22]** `filters, biases = model.layers[0].get_weights()`: pirmojo sluoksnio svoriai, forma `(3, 3, 3, 32)`
  (aukštis, plotis, įėjimo kanalai R/G/B, filtrų skaičius), ir 32 poslinkiai. Nupiešiami **2 filtrai × 3 kanalai**
  pilkais tonais (šviesu = didelis svoris).
* **[23]** Atspausdinamas vieno filtro vieno kanalo 3×3 svorių masyvas: tai tiesiog 9 skaičiai.
* **[24]** `model_layers_0 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)`: naujas modelis,
  kurio išėjimas yra **pirmojo** sluoksnio rezultatas. Jis naudoja tuos pačius išmokytus svorius.
* **[25]** `train_images[1]` paverčiamas paketu iš vienos nuotraukos (`expand_dims`), gaunami
  32 požymių žemėlapiai, forma `(1, 30, 30, 32)`.
* **[26]** Atspausdinamos nuotraukos raudonojo kanalo taškų reikšmės (32×32 skaičių lentelė), kad matytume,
  jog tinklui nuotrauka yra tik skaičiai.
* **[27]** Nupiešiami pirmieji 16 iš 32 požymių žemėlapių: skirtingi filtrai paryškina skirtingus dalykus
  (kraštus, šviesias ar tamsias sritis, tam tikras spalvas).
* **[28]–[29]** Nuotraukos forma `(32, 32, 3)`, pasirenkamas 3-iasis filtras `filters[:, :, :, 2]`, forma `(3, 3, 3)`.
* **[30]** **Konvoliucija ranka:** dviem `for` ciklais filtras „perstumiamas“ per visas 30×30 pozicijas;
  kiekvienoje sudauginami 3×3×3 nuotraukos gabalėlis ir filtras, rezultatas susumuojamas. Nupiešiamas gautas žemėlapis.
* **[31]** Nupiešiamas Keras apskaičiuotas to paties filtro žemėlapis palyginimui.
* **[32]–[33]** Pridedamas filtro poslinkis (`bias`) ir ReLU (`feature_map[feature_map < 0] = 0`), t. y. lygiai tai,
  ką daro `Conv2D(..., activation='relu')`. Dabar ranka gautas žemėlapis turi sutapti su Keras žemėlapiu.
* **[34]–[35]** Atspausdinamas abiejų žemėlapių kampas `[:6, :6]`. Abiem atvejais tai **vien nuliai**
  (tame kampe ReLU nukirpo neigiamas reikšmes), todėl šis palyginimas iš tikrųjų nieko neįrodo.
  Geriau būtų lyginti visą žemėlapį (pvz., `np.abs(a - b).max()`).

---

## 3 DALIS. Klasifikavimas daugiasluoksniu perceptronu (MLP)

* **[38]** MLP be konvoliucijų: `Flatten(input_shape=(32, 32, 3))` → 3 072 skaičiai → `Dense(32, relu)` →
  `Dense(64, relu)` → `Dense(64, relu)` → `Dense(10)`. Tas pats `compile` kaip CNN.
* **[39]** `summary()`: **105 258 parametrai**, beveik tiek pat kiek CNN. Tačiau 98 336 iš jų yra pirmajame sluoksnyje:
  kiekvienas iš 3 072 įėjimo skaičių sujungtas su kiekvienu iš 32 neuronų.
* **[41]** Mokymas 15 epochų (validavimui vėl testavimo aibė).
* **[43]–[44]** Kreivės ir `evaluate`: **tikslumas 0,4318 (~43 %)**.

### Kodėl MLP daug blogesnis

MLP nuotrauką mato kaip 3 072 nepriklausomų skaičių sąrašą: jis „nežino“, kurie taškai yra greta, ir tą patį objektą
kitoje nuotraukos vietoje turi mokytis atpažinti iš naujo. CNN filtrai ieško vietinių raštų ir tą patį filtrą taiko visur,
todėl su panašiu parametrų skaičiumi pasiekia **70 % vietoje 43 %**.

---

## Mano versija: Fashion-MNIST

`cnn_FashionMNIST.ipynb`: tos pačios trys dalys su Kaggle [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) („Zalando“ prekių nuotraukos, 28×28, nespalvotos, 10 klasių),
6 000 + 1 000 nuotraukų CSV formatu.

| | CNN | MLP |
|---|---|---|
| Tikslumas | 85,2 % | 81,1 % |

CNN vėl geresnis, bet skirtumas mažesnis nei su CIFAR-10, nes daiktas visada centre ir juodame fone.
