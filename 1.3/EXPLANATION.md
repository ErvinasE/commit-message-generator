# 1.3 užduotis: kaip veikia AdventureWorksDW sąsiuviniai

Šiame aplanke:

| Failas | Kas tai |
|---|---|
| `EXPLANATION.md` | šis paaiškinimas: kaip veikia dėstytojo sąsiuviniai, langelis po langelio |
| `Telco_Decision_Forests.ipynb` | perdarytas `AdventureWorksDW_Decision_Forests.ipynb` su Kaggle Telco duomenimis |
| `Telco_NN.ipynb` | perdarytas `AdventureWorksDW_NN.ipynb` su tais pačiais Telco duomenimis |
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Kaggle duomenų rinkinys [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

Langelių numeriai `[n]` skaičiuojami nuo 0 ir apima ir teksto, ir kodo langelius, eilės tvarka nuo sąsiuvinio pradžios.

---

## Bendra idėja

Abu sąsiuviniai sprendžia tą patį **dviejų klasių klasifikavimo** uždavinį: ar AdventureWorks klientas **nupirks dviratį**.
AdventureWorks yra „Microsoft“ sukurta išgalvota dviračių įmonė, kurios duomenų sandėlis (*DW, data warehouse*)
naudojamas SQL Server pavyzdžiuose. Rodinys `vTargetMail` buvo sukurtas tiksliniam reklaminių laiškų siuntimui:
kuriems klientams verta siųsti laišką su dviračio reklama.

* **Įvestis:** 12 kliento požymių (šeiminė padėtis, lytis, pajamos, vaikai, išsilavinimas, profesija, ar turi namą,
  kiek automobilių, kelionės į darbą atstumas, regionas, amžius);
* **Išvestis:** `BikeBuyer` (1 = pirko dviratį, 0 = nepirko);
* **Duomenų kiekis:** 18 484 klientai, 32 stulpeliai. Pirkėjų 49,4 %, t. y. klasės **subalansuotos**,
  todėl atsitiktinis spėjimas duotų ~50 % tikslumą.

Pirmasis sąsiuvinis tam naudoja **atsitiktinį sprendimų mišką** (TensorFlow Decision Forests), antrasis **neuroninį tinklą** (Keras).
Abu turi tą pačią struktūrą: 1) duomenys → lentelės, 2) lentelės → TensorFlow duomenų rinkinys, 3) modelis,
4) mokymas ir įvertinimas, 5) prognozė naujam klientui.

---

## A. `AdventureWorksDW_Decision_Forests.ipynb`

### Antraštė

* **[0]–[1]** Nuoroda į „YouTube“ vaizdo įrašą, autorius (A. Laukaitis) ir nuorodos į AdventureWorks DW schemos paveikslėlius.

### 1 žingsnis. Iš CSV failo į mokymo ir testavimo lenteles

* **[4]** `pd.read_csv(..., sep='\t')` įkelia `vTargetMail.csv` tiesiai iš GitHub. Failas atskirtas **tabuliacijos** simboliais,
  todėl reikia `sep='\t'`. Rezultatas: 18 484 eilutės × 32 stulpeliai.
* **[5]** Galimybė naudoti **savo duomenis**: jei Colab aplanke yra failas `july4.csv`, jis įkeliamas vietoj AdventureWorks
  ir `alternative_data = True`. Toliau kodas šiuo atveju elgiasi kitaip (žr. [9] ir [33]).
* **[7]** `df[0:5]` parodo pirmąsias 5 eilutes, kad matytume, kokie stulpeliai yra.
* **[9]** **Duomenų paruošimas.** Pašalinami 19 stulpelių, kurie neturi prasmės prognozei arba kartojasi:
  * identifikatoriai: `CustomerKey`, `GeographyKey`, `CustomerAlternateKey`;
  * asmeniniai duomenys: `Title`, `FirstName`, `MiddleName`, `LastName`, `NameStyle`, `Suffix`, `EmailAddress`,
    `AddressLine1`, `AddressLine2`, `Phone`;
  * vertimai: `SpanishEducation`, `FrenchEducation`, `SpanishOccupation`, `FrenchOccupation` (tas pats kaip `English...`);
  * `BirthDate` (jau yra `Age`) ir `DateFirstPurchase`.

  `BikeBuyer` pervadinamas į `target`, kad būtų galima naudoti oficialių TensorFlow pamokų kodą be pakeitimų.
  Lieka 12 požymių + `target`. Jei naudojami alternatyvūs duomenys, pašalinamas `COD`, o `Source` tampa `target`.
* **[11]** **Padalijimas 90:10.** `df.sample(frac=1)` atsitiktinai sumaišo visas eilutes, o `np.split(..., [int(0.9*len(df))])`
  perkerpa lentelę ties 90 %: 16 635 mokymo ir 1 849 testavimo eilutės. Atsitiktinių skaičių sėkla nenustatyta,
  todėl kiekvieną kartą padalijimas (ir rezultatas) šiek tiek kitoks. `FutureWarning` apie `swapaxes` yra nekenksmingas
  (`np.split` su `DataFrame` naujose versijose nebepalaikomas). Validavimo aibės nereikia (paaiškinta [10] langelyje):
  sprendimų miškas validuojasi pats „out-of-bag“ būdu (žr. [24]).

### 2 žingsnis. Iš lentelės į TensorFlow duomenų rinkinį

* **[14]** `!pip install tensorflow_decision_forests` įdiegia TF-DF biblioteką (`%%capture` paslepia diegimo išvestį).
* **[15]** `tfdf.keras.pd_dataframe_to_tf_dataset(train, label='target')` paverčia `pandas` lentelę į `tf.data.Dataset`:
  atskiria `target` nuo požymių ir suskirsto eilutes į paketus. Tekstiniai stulpeliai (`Gender`, `Region` ir kt.)
  automatiškai tampa **kategoriniais**. Sprendimų medžiams nereikia nei *one-hot* kodavimo, nei normalizavimo:
  medis tiesiog klausia „ar `Region` = `Pacific`?“ arba „ar `YearlyIncome` ≥ 55 000?“.

### 3 žingsnis. Modelis

* **[17]** `tfdf.keras.RandomForestModel(num_trees=10, max_depth=15)`. **Atsitiktinis miškas** yra daug sprendimų medžių:
  * kiekvienas medis mokomas su atsitiktine mokymo eilučių imtimi su pasikartojimais (*bootstrap*);
  * kiekviename mazge geriausias skaidymo klausimas ieškomas tik tarp atsitiktinio požymių poaibio;
  * dėl to medžiai skirtingi, o galutinė prognozė yra medžių balsavimas: tikimybė = balsavusių „taip“ medžių dalis.

  `num_trees=10`: 10 medžių (komentare minima 100); `max_depth=15`: medis gali būti iki 15 lygių gylio.

### 4 žingsnis. Mokymas ir įvertinimas

* **[20]** `model_1.fit(train_ds)`. Skirtingai nei neuroniniame tinkle, epochų nėra: kiekvienas medis sukuriamas vienu
  perėjimu per duomenis (~0,4 s). Įspėjimas apie *retracing* yra nekenksmingas.
* **[22]** `compile(metrics=['accuracy'])` ir `evaluate(test_ds)`: **tikslumas su testavimo aibe 0,8091 (~81 %)**.
  `loss: 0.0000e+00` rodomas todėl, kad TF-DF modeliai Keras nuostolio neskaičiuoja.
* **[24]** **Mokymo žurnalas.** `make_inspector().training_logs()` grąžina kiekvieno miško dydžio (1, 2, …, 10 medžių)
  **out-of-bag** (OOB) įvertinimą. OOB: kiekvienam medžiui apie trečdalis mokymo eilučių nepateko į jo imtį,
  todėl tos eilutės tam medžiui yra „nematyti“ duomenys. Grafikai rodo, kaip didėjant medžių skaičiui kyla OOB
  tikslumas ir krenta *logloss* nuostolis.
* **[26]** `tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=15)` nupiešia interaktyvų
  pirmąjį (`tree_idx=0`) miško medį: kiekviename mazge yra klausimas, o lapuose klasių pasiskirstymas.

### 5 žingsnis. Prognozavimas

* **[29]** `model_1.save(...)` išsaugo modelį TensorFlow `SavedModel` aplanku. Pavadinime parašyta `trees300`,
  bet modelis turi 10 medžių (pavadinimas likęs nuo ankstesnio bandymo).
* **[31]** `tf.keras.layers.TFSMLayer(...)` įkelia išsaugotą modelį (Keras 3 būdas įkelti `SavedModel` tik prognozėms).
  Įkeltas `reloaded_model` toliau **nenaudojamas** ([37] eilutė užkomentuota).
* **[33]–[34]** Naujas klientas aprašomas žodynu su visais 12 požymių (vedęs vyras, 42 m., pajamos 70 000,
  2 vaikai, bakalauras, profesionalas, turi namą, 1 automobilis, Ramiojo vandenyno regionas, 1–2 mylios iki darbo).
  Jei naudojami alternatyvūs duomenys, imama pirmoji lentelės eilutė.
* **[36]** Žodynas → `DataFrame` → TF duomenų rinkinys → `model_1.predict`. Rezultatas `[[0.9]]`: **90 % tikimybė,
  kad klientas pirks dviratį**. Kadangi medžių yra 10, tikimybė visada yra 0,1 kartotinis (9 iš 10 medžių balsavo „taip“).

### Papildoma dalis: `dtreeviz`

* **[39]–[40]** Įdiegiama ir importuojama `dtreeviz` biblioteka gražesniems medžių paveikslėliams.
* **[42]** Kodas, kuris turėjo sukurti vizualizaciją `viz_cmodel`, yra **įdėtas į kabutes `'''...'''`**, t. y. paverstas
  tekstu ir nevykdomas. Todėl **[43]** ir **[44]** langeliai baigiasi klaida `NameError: name 'viz_cmodel' is not defined`.
  Šie langeliai turėjo parodyti medį su kiekvieno mazgo duomenų histogramomis ([43]) ir vieno kliento
  (mokymo eilutė nr. 20) kelią per medį ([44]).

---

## B. `AdventureWorksDW_NN.ipynb`

### 1 žingsnis. Iš CSV failo į mokymo, validavimo ir testavimo lenteles

* **[4]–[7]** Tas pats `vTargetMail.csv` įkėlimas (į `df_all_col`), visos lentelės ir vienos eilutės (nr. 10) peržiūra.
* **[9]** Tie patys 19 stulpelių pašalinami, `BikeBuyer` → `target` (kaip A dalies [9]).
* **[10]** Atspausdinami stulpelių **tipai**. Neuroniniam tinklui tai svarbu: `object` (tekstas) bus koduojamas kaip
  kategorija, `int64` skaičiai bus normalizuojami. Pastaba: `Age` apskaičiuotas pagal dabartinę datą, todėl
  klientų amžius yra 42–112 metų.
* **[12]** **Padalijimas 80:10:10**: 14 787 mokymo, 1 848 validavimo, 1 849 testavimo eilutės. Neuroniniam tinklui
  **validavimo** aibė reikalinga, kad po kiekvienos epochos matytume, ar tinklas nepersimoko (*overfitting*).

### 2 žingsnis. Iš lentelės į TensorFlow duomenų rinkinį

* **[14]** Funkcija `df_to_dataset`:
  1. nukopijuoja lentelę ir išima `target` (`labels = df.pop('target')`);
  2. kiekvieną stulpelį paverčia `(N, 1)` formos masyvu ir sudeda į žodyną `{stulpelio pavadinimas: masyvas}`;
  3. `tf.data.Dataset.from_tensor_slices((žodynas, labels))` sukuria duomenų rinkinį;
  4. `shuffle` sumaišo eilutes (tik mokymo aibei), `batch(256)` suskirsto į paketus po 256, `prefetch` paruošia
     kitą paketą, kol skaičiuojamas dabartinis.

  **Klaida:** žodynas kuriamas iš `dataframe.items()` (originalios lentelės), o ne iš `df.items()` (kopijos be `target`).
  Todėl `target` patenka tarp įėjimų. Modelis jo nenaudoja, nes tokio įėjimo neturi, bet [28] langelyje dėl to
  rodomas ilgas įspėjimas `The structure of inputs doesn't match the expected structure ... 'target'`.

### 3 žingsnis. Modelis

Modelis kuriamas Keras **funkciniu** būdu: kiekvienam požymiui atskiras įėjimas (`keras.Input`),
jo kodavimo sluoksnis, tada viskas sujungiama.

* **[16]** `all_inputs` (žodynas įėjimų) ir `encoded_features` (sąrašas užkoduotų požymių).
* **[18]** **Kategoriniai stulpeliai.** `get_category_encoding_layer`:
  1. `StringLookup` (tekstui) arba `IntegerLookup` (sveikiesiems skaičiams) su `adapt` peržiūri mokymo duomenis ir
     kiekvienai reikšmei priskiria numerį (0 rezervuotas nežinomoms reikšmėms);
  2. `CategoryEncoding` numerį paverčia *one-hot* vektoriumi (pvz., `Region`: `Europe` → `[0, 0, 1, 0]`).

  Taip koduojami `MaritalStatus`, `Gender`, `EnglishEducation`, `EnglishOccupation`, `CommuteDistance`, `Region`
  (tekstas) ir `HouseOwnerFlag` (0/1, nors tai skaičius, prasmė yra kategorija).
* **[20]** **Skaitiniai stulpeliai.** `Normalization(axis=None)` su `adapt` apskaičiuoja vidurkį ir dispersiją, o vėliau
  kiekvieną reikšmę perskaičiuoja: (x − vidurkis) / standartinis nuokrypis. Taip `YearlyIncome` (10 000–170 000)
  neužgožia `TotalChildren` (0–5). Normalizuojami `YearlyIncome`, `TotalChildren`, `NumberChildrenAtHome`, `NumberCarsOwned`, `Age`.
* **[21]** `concatenate` sujungia visus užkoduotus požymius į vieną vektorių.
* **[23]** **Tinklas:**
  * `Dense(64, relu)` → `Dropout(0.2)` → `Dense(32, relu)` → `Dropout(0.3)` → `Dense(1)`;
  * `Dense` neuronas apskaičiuoja svertinę įėjimų sumą + poslinkį; `relu` neigiamas reikšmes paverčia 0;
  * `Dropout` mokymo metu atsitiktinai išjungia 20 % / 30 % neuronų, kad tinklas nepersimokytų;
  * paskutinis `Dense(1)` be aktyvacijos grąžina **logitą** (bet kokį realųjį skaičių);
  * `compile`: optimizatorius `adam`, nuostolis `BinaryCrossentropy(from_logits=True)` (dviejų klasių uždaviniui,
    sigmoidė pritaikoma nuostolio viduje), metrika `accuracy`.
* **[24]** Užkomentuota alternatyva: vienas `Dense(1)` sluoksnis tiesiai ant požymių, t. y. logistinė regresija.
* **[25]** `plot_model` nupiešia modelio schemą: 12 įėjimų → kodavimo sluoksniai → `Concatenate` → `Dense` sluoksniai.

### 4 žingsnis. Mokymas ir įvertinimas

* **[28]** `model.fit(train_ds, validation_data=validation_ds, epochs=30)`: 30 epochų, kiekvienoje 58 paketai po 256 eilutes.
  Tikslumas kyla nuo ~0,56 (1 epocha) iki ~0,70 (7 epocha) ir toliau.
* **[30]** `model.evaluate(test_ds)`: **tikslumas su testavimo aibe 0,7474 (~75 %)**, t. y. mažesnis nei sprendimų miško (~81 %).

### 5 žingsnis. Prognozavimas

* **[33]** `model.save('...keras')` išsaugo visą modelį (architektūrą, svorius, kodavimo žodynus) vienu failu.
* **[35]** `tf.keras.models.load_model` jį įkelia.
* **[37]** Tas pats naujas klientas kaip A dalyje.
* **[39]** Kiekviena reikšmė paverčiama tenzoriumi, `predict` grąžina logitą, `tf.nn.sigmoid` jį paverčia tikimybe:
  **97,7 % tikimybė pirkti dviratį**.

### TensorBoard

* **[41]–[42]** Įkeliamas TensorBoard plėtinys, ištrinami seni žurnalai (`!rm -rf ./logs/`).
* **[43]** `TensorBoard` *callback* kiekvienos epochos metrikas įrašo į `logs/fit/<data-laikas>`. **Svarbu:** `model.fit`
  čia **tęsia jau išmokyto modelio mokymą** dar 40 epochų, o ne pradeda iš naujo. Komentaras sako, kad `histogram_freq=1` sukelia klaidą.
* **[44]** `%tensorboard --logdir logs/fit` atidaro interaktyvų TensorBoard langą su mokymo ir validavimo kreivėmis.

### „Tmp“ dalis

* **[46]** Prognozė tam pačiam klientui po papildomų 40 epochų: 87,98 % (anksčiau 97,7 %). Skiriasi, nes modelis buvo mokomas toliau.
* **[47]** Atspausdina paruoštą lentelę `df`.
* **[48]** `eval_js` pakeičia Colab išvesties lango aukštį (techninis Colab nustatymas).

---

## Palyginimas

| | Sprendimų miškas | Neuroninis tinklas |
|---|---|---|
| Tikslumas su testavimo aibe | ~81 % | ~75 % |
| Duomenų paruošimas | nereikia (kategorijas atpažįsta pats) | kiekvienam stulpeliui kodavimo / normalizavimo sluoksnis |
| Mokymo laikas | <1 s | ~30 epochų |
| Validavimo aibė | nereikia (OOB) | reikia |
| Paaiškinamumas | galima nupiešti medį | „juoda dėžė“ |

Šiems lentelės pavidalo duomenims sprendimų miškas buvo ir tikslesnis, ir paprastesnis.

---

## Mano versija: Telco klientų nutekėjimas

Abu sąsiuviniai perdaryti su Kaggle duomenų rinkiniu **Telco Customer Churn** (IBM pavyzdiniai duomenys):
7 043 telekomunikacijų bendrovės klientai, klausimas: **ar klientas išeis** (`Churn`).
Uždavinys tokio pat tipo kaip AdventureWorks (vienas klientas = viena eilutė, tekstiniai ir skaitiniai požymiai,
atsakymas taip/ne), todėl visi originalų žingsniai tinka.

| Kas pakeista | Kodėl |
|---|---|
| 12 požymių: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `InternetService`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | tiek pat kaip originale; pašalintas ID ir 7 papildomų paslaugų stulpeliai |
| `TotalCharges` tekstas → skaičius, 11 tuščių → 0 | CSV faile 11 naujų klientų šis laukas tuščias |
| Rodomas **bazinis lygis** (~76 %) ir klaidų matrica | išėjusių tik 26,5 %, todėl tikslumas be bazinio lygio klaidina |
| TF-DF → **YDF** | `tensorflow_decision_forests` neįsidiegia su Colab Python 3.13; YDF yra ta pati „Google“ biblioteka be TensorFlow sluoksnio |
| `random_state=42` / `set_random_seed(42)` | atkartojami rezultatai; abiejų sąsiuvinių testavimo aibė ta pati |
| Pataisytas `df_to_dataset` | `target` nebepatenka tarp įėjimų |
| Pridėtos mokymo kreivės, požymių svarba | rezultatams paaiškinti |
| Praleista `dtreeviz` dalis | originale neveikė, YDF jos nepalaiko; medis piešiamas su YDF `plot_tree` |

**Rezultatai** (ta pati testavimo aibė, 705 klientai):

| | Sprendimų miškas | Neuroninis tinklas |
|---|---|---|
| Tikslumas | 79,3 % | 80,9 % |
| Bazinis lygis | 75,7 % | 75,7 % |
| Atpažinta išėjusių | 81 iš 171 | 85 iš 171 |

Skirtumas tarp modelių yra apie 11 klientų, t. y. per mažas, kad būtų galima teigti, jog vienas modelis geresnis.
Svarbiausi požymiai yra sutarties tipas, kliento stažas ir interneto tipas. Išsamiau žr. sąsiuvinių „Išvadas“.
