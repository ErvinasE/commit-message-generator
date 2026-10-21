# 1.5 užduotis: kaip veikia `cnn_One_Image.ipynb` ir `cnn_Transfer_learning.ipynb`

Šiame aplanke:

| Failas | Kas tai |
|---|---|
| `EXPLANATION.md` | šis paaiškinimas: kaip veikia abu dėstytojo sąsiuviniai, langelis po langelio |
| `cnn_One_Image_OpenImages.ipynb` | perdarytas `cnn_One_Image.ipynb` su Open Images verslo aplinkos nuotraukomis |
| `cnn_Transfer_learning_FashionMNIST.ipynb` | perdarytas `cnn_Transfer_learning.ipynb` su Fashion-MNIST prekių nuotraukomis |
| `data/images/*.jpg`, `data/images.csv` | 6 Open Images nuotraukos (CC BY 2.0) ir jų autoriai / licencijos |
| `data/oid_v4_labels.csv` | objektų detektoriaus 600 klasių pavadinimai (Open Images V4) |
| `data/ade20k_labels.csv` | segmentavimo modelio 150 klasių pavadinimai (ADE20K) |
| `data/fashion-mnist_*.csv.gz` | tie patys 6 000 + 1 000 Fashion-MNIST nuotraukų kaip 1.4 užduotyje |

Langelių numeriai `[n]` skaičiuojami nuo 0 ir apima ir teksto, ir kodo langelius.

Abu sąsiuviniai naudoja **iš anksto išmokytus** (*pre-trained*) tinklus: kažkas jau išmokė didelį tinklą su milijonais
nuotraukų, o mes jį panaudojame. Pirmajame sąsiuvinyje tinklai naudojami tokie, kokie yra (be mokymo),
antrajame vienas toks tinklas pritaikomas naujai užduočiai (*transfer learning*).

---

## A. `cnn_One_Image.ipynb`: klasifikavimas, objektų aptikimas ir segmentavimas su viena nuotrauka

Sąsiuvinis paima vieną nuotrauką (pvz., Vilniaus Šv. Onos bažnyčią) ir paleidžia ją per tris skirtingų tipų modelius
iš **TensorFlow Hub**:

| Užduotis | Klausimas | Atsakymas |
|---|---|---|
| Klasifikavimas | kas yra nuotraukoje? | klasių sąrašas su tikimybėmis visai nuotraukai |
| Objektų aptikimas | kas ir kur? | stačiakampiai su klase ir tikimybe |
| Segmentavimas | kuriai klasei priklauso kiekvienas taškas? | klasės žemėlapis kiekvienam taškui |

### 1 skyrius. Vaizdų klasifikavimas (2 modeliai)

* **[0]–[1]** Antraštė ir nuoroda į vaizdo įrašą.
* **[2]** Importai: `tensorflow`, `tensorflow_hub` (modelių įkėlimui), `requests` ir `BytesIO` (nuotraukoms iš interneto),
  PIL (piešimui ant nuotraukų), `matplotlib`, `HTML` (gražiam atsakymų rodymui).
* **[3]** Pagalbinės funkcijos:
  * `preprocess_image`: PIL nuotrauka → `numpy` → forma `[1, aukštis, plotis, kanalai]` (paketas iš 1 nuotraukos) →
    `float32` reikšmės 0–1 (`convert_image_dtype`);
  * `load_image_from_url`: parsisiunčia nuotrauką (su `User-agent` antrašte, nes Wikimedia atmeta užklausas be jos);
  * `load_image`: grąžina dvi versijas: `img_raw` (originalaus dydžio) ir `img`, padidintą arba sumažintą iki
    `image_size × image_size` su `resize_with_pad` (proporcijos išlaikomos, trūkstama dalis užpildoma juodomis juostomis).
    Jei nuotrauka nespalvota, jos kanalas nukopijuojamas 3 kartus;
  * `show_image`: parodo nuotrauką;
  * `run_model_on_image`: paleidžia klasifikatorių, logitus `tf.nn.softmax` paverčia tikimybėmis, `argsort` surikiuoja
    ir grąžina 4 labiausiai tikėtinas klases kaip tekstą;
  * `load_img`: nenaudojama.
* **[4]** Įkeliami 2 klasifikatoriai su `hub.load`:
  * **Modelis 1**: `efficientnet_v2_imagenet21k_s`, EfficientNetV2-S, išmokytas su **ImageNet-21k** (~14 mln. nuotraukų,
    21 843 klasės). Klasių pavadinimai iš `imagenet21k_wordnet_lemmas.txt`;
  * **Modelis 2**: `efficientnet_v2_imagenet21k_ft1k_s`, tas pats tinklas, išmokytas su ImageNet-21k ir paskui
    **suderintas** (*fine-tuned*) su ImageNet-1k (1 000 klasių). Pavadinimai iš `ImageNetLabels.txt`, pirmoji eilutė
    (`background`) išmetama su `[1:]`;
  * abiejų įėjimo dydis 384×384. Užkomentuoti didesni „M“ modeliai (480×480) būtų 1–2 % tikslesni, bet reikalautų dvigubai daugiau atminties.
* **[6]** **Pasirinkimo forma** (`# @param` sukuria Colab išskleidžiamąjį sąrašą, `run: "auto"` langelį paleidžia iš naujo
  pakeitus pasirinkimą). Žodynas sieja pavadinimus su Wikimedia nuorodomis. Nuotrauka įkeliama, parodoma ir paleidžiama per abu modelius;
  atsakymai rodomi HTML: mėlyna modelis 1, žalia modelis 2.
* **[8]** **Iškarpa.** Slankikliai `x`, `y` (procentais) nurodo iškarpos viršutinį kairįjį kampą, `cut_size_in_pixels`
  (500) jos dydį. Koordinatės skaičiuojamos nuo originalaus dydžio nuotraukos (`original_image.shape[2]` yra plotis,
  `shape[1]` aukštis, nes `shape[0]` yra paketo dimensija). Iškarpa padidinama iki 384×384 su `tf.image.resize`
  (be juostų, todėl ne kvadratinė iškarpa ištempiama) ir klasifikuojama iš naujo. Taip matyti, kad klasifikatorius
  atsako apie **visą** kadrą: iškirpus detalę, atsakymas pasikeičia. **Svarbu:** kintamasis `image` čia perrašomas iškarpa,
  todėl 2 ir 3 skyriuose `image` jau yra iškarpa.

### 2 skyrius. Objektų aptikimas

* **[10]** Įkeliamas detektorius `openimages_v4/ssd/mobilenet_v2`: **SSD** (*Single Shot Detector*) su MobileNetV2
  pagrindu, išmokytas su **Open Images V4** (600 objektų klasių). `signatures['default']` yra modelio funkcija prognozei.
  Užkomentuotas Faster R-CNN Inception-ResNet V2 būtų tikslesnis, bet reikalautų daugiau atminties. Atsisiunčiamas ir
  klasių aprašų CSV, bet jo nereikia: modelis klasių pavadinimus grąžina pats (`detection_class_entities`).
* **[11]** Pagalbinės funkcijos (iš oficialios TF Hub pamokos):
  * `display_image`: parodo nuotrauką dideliu formatu;
  * `download_and_resize_image`: nenaudojama (ir neveiktų, nes `tempfile` ir `urlopen` neimportuoti);
  * `draw_bounding_box_on_image`: nupiešia stačiakampį ir virš jo spalvotą juostelę su tekstu (jei virš stačiakampio
    vietos nėra, tekstas dedamas žemiau);
  * `draw_boxes`: pereina per 10 (čia 20) labiausiai tikėtinų radinių, kurių tikimybė ≥ `min_score` (0,1),
    ir kiekvienam nupiešia stačiakampį; spalva parenkama pagal klasės pavadinimo maišos (*hash*) reikšmę;
  * `run_detector_2`: paleidžia detektorių (įėjimas `float32` 0–1), rezultatus paverčia `numpy`, atspausdina radinių skaičių
    ir nupiešia stačiakampius. Radinių visada 100, nes modelis visada grąžina 100 labiausiai tikėtinų, net ir labai mažos tikimybės.
* **[13]** Detektorius originalaus dydžio nuotraukoje.
* **[14]** Detektorius iškarpoje (`image`).

### 3 skyrius. Vaizdo segmentavimas

* **[16]** Įkeliamas **HRNet** segmentavimo modelis, išmokytas su **ADE20K** (150 klasių: siena, pastatas, dangus,
  grindys, medis, kelias, žmogus ir t. t.). Įspėjimai apie *custom gradients* nekenksmingi (gradientų nereikia, nes nemokome).
  Komentaruose minima, kad COCO ir Cityscapes HRNet versijos neatitinka dokumentacijos.
* **[17]** Klasių pavadinimai iš Hugging Face `ade20k-id2label.json`; priekyje pridedamas `background`, iš viso 151.
* **[18]** `segmentation_ade20k_model.predict(image)` grąžina formos `(1, aukštis, plotis, 151)` masyvą: kiekvienam
  taškui 151 tikimybę. Nupiešiami visi 151 tikimybių žemėlapiai 13×13 tinklelyje (šviesu = didelė tikimybė).
* **[19]** 4 pasirinktų klasių (`[2, 5, 13, 18]` = pastatas, medis, žmogus, augalas) žemėlapiai uždedami ant nuotraukos
  pusiau permatomai (`alpha=0.5`).

---

## B. `cnn_Transfer_learning.ipynb`: perkeliamasis mokymasis su CIFAR-10

**Idėja:** didelis tinklas (EfficientNetV2-B1), išmokytas su ImageNet-21k, jau moka išskirti bendrus vaizdo požymius
(kraštus, tekstūras, formas). Jo paskutinis klasifikavimo sluoksnis nuimamas, o likusi dalis naudojama kaip
**požymių išskyrėjas** (*feature extractor*): kiekvienai nuotraukai grąžina 1 280 skaičių vektorių.
Ant jo uždedamas naujas `Dense(10)` sluoksnis CIFAR-10 klasėms.

### 1 skyrius. Modelio sukūrimas, mokymas ir išsaugojimas

* **[1]** Importai, tarp jų `tensorflow_hub`.
* **[2]** Paaiškinimas: šį skyrių reikia vykdyti, jei išsaugoto modelio (`saved_model_CIFAR10.zip`) dar nėra.
* **[5]** CIFAR-10 įkeliamas su `datasets.cifar10.load_data()`, taškai padalijami iš 255 (0–1).
* **[6]** `hub.KerasLayer(".../efficientnet_v2_imagenet21k_b1/feature_vector/2", trainable=False)`: požymių išskyrėjas,
  įėjimas 240×240×3, išėjimas 1 280 skaičių. `trainable=False` reiškia, kad jo svoriai **užšaldyti**.
* **[7]–[8]** Įdiegiamas `tf_keras` (senasis Keras 2), nes `hub.KerasLayer` neveikia su Keras 3. Modelis:
  1. `Resizing(240, 240)`: 32×32 nuotrauka padidinama iki 240×240 (tokio dydžio nuotraukomis tinklas išmokytas);
  2. `feature_extractor_layer`: EfficientNetV2-B1;
  3. `Dense(10)`: nauji klasifikatoriaus svoriai.

  `compile`: `adam`, `SparseCategoricalCrossentropy(from_logits=True)`, `accuracy`.
* **[9]** Užkomentuota Keras 3 versija, kuri neveikia su `hub.KerasLayer`.
* **[11]** `build` ir `summary`: iš viso 6 943 934 parametrai, bet **mokomi tik 12 810** (1 280 · 10 + 10, tai `Dense` sluoksnis);
  6 931 124 EfficientNet parametrai užšaldyti.
* **[13]** `model.fit(..., epochs=20, validation_data=(test_images, test_labels))`: mokomas tik paskutinis sluoksnis
  (validavimui vėl naudojama testavimo aibė).
* **[14]** `evaluate`: **tikslumas 0,8923 (~89 %)**. Palyginimui: CNN nuo nulio (`cnn.ipynb`) pasiekė ~70 %.
* **[16]** Modelis išsaugomas `SavedModel` aplanku ir suspaudžiamas į ZIP (kad būtų galima parsisiųsti ir vėliau įkelti).
* **[18]** Užkomentuotas vienos nuotraukos prognozės pavyzdys.
* **[19]** Prognozės pirmosioms 100 testavimo nuotraukų, po vieną (`%%capture` paslepia 100 eilučių išvesties).
* **[20]** 100 nuotraukų tinklelis su užrašu `tikroji klasė/prognozė`: žalia, jei teisingai, raudona, jei klaida.
  (Antraštė [17] „Let's plot the first 100 images“ šiek tiek klaidina: rodomos prognozės.)

### 2 skyrius. Duomenų papildymas ir viso tinklo derinimas (*fine-tuning*)

* **[22]–[24]** Iš naujo įkeliamas CIFAR-10, išskleidžiamas ZIP, įkeliamas modelis su `tf_keras.models.load_model`,
  laikini failai ištrinami.
* **[25]** Atspausdinama, kurie sluoksniai mokomi. Po įkėlimo visi trys rodo `Trainable: True`.
* **[26]** Mokymosi greitis sumažinamas 10 kartų (0,001 → 0,0001): derinant jau išmokytus svorius reikia mažų žingsnių,
  kad jie nebūtų sugadinti.
* **[27]** `model.trainable = True`: dabar mokomi **visi** ~6,9 mln. parametrų, ne tik paskutinis sluoksnis.
  (Keras rekomenduoja po `trainable` pakeitimo iš naujo iškviesti `compile()`; čia to nėra.)
* **[28]** **Duomenų papildymas** (*data augmentation*): `RandomFlip('horizontal')` (atspindys), `RandomRotation(0.1)`
  (pasukimas iki ±36°), `RandomBrightness(0.1)` (šviesumas), `RandomTranslation(0.1, 0.1)` (poslinkis iki 10 %).
  Užkomentuoti kiti variantai (aukštis, plotis, kontrastas, priartinimas). Papildymas sukuria „naujas“ nuotraukas
  iš senų, todėl tinklas mažiau persimoko.
* **[29]** Papildymas pritaikomas visam mokymo masyvui vieną kartą.
* **[30]** Ciklas `for i in range(100)`: kiekvieną kartą iš naujo įkeliami duomenys, pritaikomas naujas atsitiktinis
  papildymas ir mokoma 2 epochas. Iš viso 200 epochų po ~122 s su GPU, t. y. **~7 valandos**. Matomoje išvestyje
  validavimo tikslumas pirmosiose iteracijose yra ~0,87, t. y. **mažesnis** nei 0,89 prieš derinimą: iš pradžių papildyti
  duomenys ir viso tinklo derinimas rezultatą pablogina. Komentare užrašyti skaičiai 0,9537 ir 0,9406, matyt,
  yra ilgo mokymo rezultatas.
* **[31]** Nupiešiamos 100 papildytų mokymo nuotraukų (matyti pasukimai, poslinkiai, atspindžiai).
* **[32]–[34]** Patikrinimai: `model.layers[2].trainable`, visų svorių pavadinimų sąrašas (EfficientNet blokai) ir
  sluoksnių skaičius: **3**, nes visas EfficientNet yra vienas `KerasLayer` sluoksnis.

---

## Mano versijos

### `cnn_One_Image_OpenImages.ipynb`

Vietoj Wikimedia nuotraukų naudojamos **6 verslo aplinkos nuotraukos** iš „Google“ duomenų rinkinio
[Open Images](https://storage.googleapis.com/openimages/web/index.html) (validavimo dalis, licencija CC BY 2.0):
kavinė iš lauko, biuras, parduotuvė, vaisių prekystalis, gatvė su transportu, kavinė viduje. Nuotraukos saugomos
`data/images/`, o jų autoriai ir nuorodos yra `data/images.csv`. Savo nuotraukas galima pridėti į tą patį aplanką ir CSV.

| Kas pakeista | Kodėl |
|---|---|
| Klasifikatoriai: `keras.applications` **EfficientNetV2-S** (384×384) ir **MobileNetV3-Large** (224×224), abu ImageNet-1k | palyginimas „didelis ir tikslus“ prieš „mažas ir greitas“; nereikia `tensorflow_hub` |
| Detektorius: **tas pats** SSD MobileNet V2 Open Images V4, bet parsisiunčiamas kaip `SavedModel` archyvas iš TensorFlow modelių saugyklos | TF Hub dabar nukreipia į Kaggle; klasių pavadinimai iš `data/oid_v4_labels.csv` |
| Segmentavimas: **DeepLabV3 MobileNetV2**, **tos pačios** ADE20K 150 klasių | HRNet pasiekiamas tik per TF Hub; DeepLab yra TF1 „užšaldytas grafas“, įkeliamas su `tf.compat.v1.wrap_function` |
| Papildomai: klasifikavimo lentelė ir detektorius visoms 6 nuotraukoms | matyti, kaip modeliai veikia su visu nuotraukų rinkiniu |
| 16 didžiausių klasių žemėlapiai vietoj visų 151; 4 uždedamos klasės parenkamos automatiškai; spalvotas segmentavimo žemėlapis | įskaitomumas; veikia su bet kuria nuotrauka |
| Segmentavimas atliekamas visai nuotraukai, ne iškarpai | originale `image` buvo perrašytas iškarpa |

**Rezultatai (gatvės nuotrauka):**
* klasifikavimas: abu modeliai sako `cab` (76 % / 80 %), iškarpoje su motoroleriu `moped` (77 % / 34 %);
* aptikimas: 2 automobiliai, taksi, dviratis, žmogus, ratas; autobusų ir daugumos pėsčiųjų nerado;
  vaisių prekystalyje nerado nieko (modelis pats atmeta radinius su tikimybe < 30 %);
* segmentavimas: automobilis 30 %, pastatas 20 %, medis 17 %, kelias 13 %, žmogus 5 % ploto; didelę autobuso dalį
  modelis pažymėjo kaip automobilį.

Mažesnis klasifikatorius ir greiti „mobilūs“ detekcijos / segmentavimo modeliai daro pastebimai daugiau klaidų nei dideli.

### `cnn_Transfer_learning_FashionMNIST.ipynb`

Naudojami tie patys 6 000 + 1 000 **Fashion-MNIST** prekių nuotraukų kaip 1.4 užduotyje, todėl perkeliamąjį mokymąsi
galima tiesiogiai palyginti su tinklu, mokytu nuo nulio.

| Kas pakeista | Kodėl |
|---|---|
| `keras.applications.EfficientNetV2B1` (ImageNet-1k) vietoj TF Hub `KerasLayer` (ImageNet-21k) | tas pats tinklas, veikia su Keras 3 be `tf_keras` |
| 1 kanalas nukopijuojamas į 3, taškai paliekami 0–255 | EfficientNet tikisi spalvotų nuotraukų; Keras versija pati normalizuoja |
| `validation_split=0.1` vietoj testavimo aibės | testavimo aibė lieka tik galutiniam įvertinimui |
| 5 epochos (užšaldytas) + 3 epochos (derinimas) vietoj 20 + 200 | pakanka parodyti efektą; originalo 200 epochų truko ~7 val. |
| Derinant `BatchNormalization` sluoksniai lieka užšaldyti, modelis iš naujo kompiliuojamas | Keras rekomendacija; kitaip sugadinama ImageNet statistika |
| Papildymas per `tf.data` kiekvienam paketui | kiekvienoje epochoje kitokios nuotraukos, nereikia 100 kartų įkelti duomenų |
| `.keras` failas vietoj `SavedModel` + ZIP | Keras 3 formatas |

**Rezultatai (1 000 testavimo nuotraukų):**

| Modelis | Tikslumas |
|---|---|
| CNN nuo nulio (1.4) | 85,2 % |
| EfficientNetV2-B1 užšaldytas | 87,7 % |
| + derinimas su papildytais duomenimis | **89,1 %** |

Kaip ir originale, pirmoje derinimo epochoje validavimo tikslumas nukrito (87 % → 80 %), o vėliau pakilo aukščiau
pradinio (88,5 %). Pagerėjimas mažesnis nei su CIFAR-10, nes 28×28 nespalvotos nuotraukos labai skiriasi nuo ImageNet nuotraukų.
