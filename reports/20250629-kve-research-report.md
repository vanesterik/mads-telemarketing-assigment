---
date: 29th of June 2025
title: Predicting Telemarketing Success Rate for Banking Product Subscriptions
subtitle: Predictive Modelling (2024 P3A)
institute: HAN - Master Applied Data Science
author: |
  | Koen van Esterik
  | kd.vanesterik@student.han.nl
---

# 0. Abstract



# 1. Introduction

Voor elk commercieel bedrijf bestaat de behoefte om winst te maken. Dit om op basis daarvan investeringen te doen, groei te realiseren en uiteindelijk bestaanszekerheid te garanderen. Naast deze behoefte bestaat ook een verplichting richting aandeelhouders. Deze verplichting bestaat uit het creeëren van waarde om daarmee de aandelen van de aandeelhouders te vergroten.

Dit geldt ook voor investeringsmaatschappijen, die als voornaamste activitieit, investeren in andere bedrijven om daarbij waarde te creeëren en winst te genereren. Het succes van een investeringsmaatschappij wordt bepaalt door het vermogen om de juiste investeringsbeslissingen te nemen en daarmee de waarde van aandelen van aandeelhouders te maximaliseren.

Met dit als context richten we onze aandacht op de investeringsmaatschappij Blackrock, die een investering overweegt - om een Portugese bank over te nemen. Deze Portugese bank voert, naast reguliere financiële activeitien, telemarketing campagnes uit om financiële producten te verkopen.

De meest robuuste telemarketing operatie bestaat uit campagnes om abonementen voor bank depositos te werven. Ze gebruiken hiervoor gegevens van bestaande klanten om naar uit te bellen. Dit met wisselend succes, omdat het moeilijk te voorspellen is - hoe een klant gaat reageren op een mogelijk storend telefoongesprek. Veel van deze gesprekken zijn geregistreerd in een dataset, waarin is aangegeven of de prospect een bank deposito abonnement wil afnemen.

Hier ligt voor Blackrock een mogelijkheid om dieper strategisch inzicht te verkrijgen met betrekking tot de overname van de Portugese bank. Dit door te onderzoeken of winst gemaximaliseerd kan worden, door het telemarketing process te optimaliseren door een prospect selectieprocedure te implementeren.

De bijgehouden dataset kan dienen als input voor het onderzoek. Dit onderzoek zal gefocust zijn op de gesuggereerde selectieprocedure. Deze proceducure wordt gebasseerd op een machine learning voorspellingsmodel. Dit voorspellingsmodel kan inzicht bieden, die Blackrock kan gebruiken om mee te laten wegen in de overname van de Portugese bank.

Dit onderzoek laat zich verwoorden als:

- Reduceer willekeurig gekozen telemarketing campagne prospects,
- Door de ontwikkeling van een selectieprocedure op basis van een voorspellingsmodel,
- Die beter presteert dan het uitbellen naar alle prospects,
- Om winstmaximalisatie te realiseren

Het voorspellingsmodel is gebasseerd op binaire classificatie, want we willen voorspellen of een prospect een bank deposito abonnement zal afnemen - ja of nee. Deze informatie is aanwezig in de eerder genoemde dataset en zal als trainings- en test-data dienen.

Binaire classificatie bestaat uit een begeleide leermethode om gegevens te categoriseren in één van twee mogelijke resultaten. Dit om op basis hiervan voorspellingen te doen op nieuwe, ongeziene gegevens. Bij het evalueren van de prestaties van het binaire classificatiemodel worden de volgende termen gebruikt:

- True Positive (TP): Het model voorspelt correct een positieve uitkomst.
- False Negative (FN): Het model voorspelt incorrect een negatieve uitkomst.
- False Positive (FP): Het model voorspelt incorrect een positieve uitkomst.
- True Negative (TN): Het model voorspelt correct een negatieve uitkomst.

Deze termen worden gebruikt om verschillende standaard metrics uit te rekenen - bijv. accuracy, precision, recall, etc. Deze metrics dienen om het classificatiemodel te evalueren. In dit onderzoek zijn deze termen van belang om verschillende classificatiemodellen met elkaar te vergelijken, om te beoordelen welke de meeste winst oplevert. Daarbij stellen we een eigen metric voor die de meeste winst berekent op basis van de eerder genoemde termen (TP, FN, FP en TN). We noemen deze metric de **Maximum Profit** (MP) metric.

De MP metric introduceren we, om een verbinding te leggen tussen het technische gedeelte van het onderzoek en de business case. Dit met de gedachte dat standaard metrics vaak onvoldoende inzicht bieden met betrekking tot de strategische beslissingen die Blackrock doorgaans moet maken.

Om de maximale winst van een classificatiemodel uit te rekenen, stellen we de volgende formules voor:

$$
\vec{p} = \textit{r} * \vec{tps} - \textit{c} * (\vec{tps} + \vec{fps})
$$

waarbij:

- $\vec{p}$, vector met de winst voor alle drempelwaarden;
- $\textit{r}$, scalar met de opbrengst per succesvol gesprek;
- $\textit{c}$, scalar met de kosten per gesprek;
- $\vec{tps}$, vector met TP's voor alle drempelwaarden;
- $\vec{fps}$, vector met FP's voor alle drempelwaarden;

De bovenstaande formule geeft de input om de maximale winst vast te stellen, door de formule:

$$
\textit{p} = \max{(\vec{p})}
$$

waarbij:

- $\textit{p}$, scalar met de maximale winst;
- $\vec{p}$, vector met de winst voor alle drempelwaarden;

De MP metric zal uitkomst bieden bij het evalueren van alle mogelijke classificaitemodellen en zal een uiteindelijke model selecteren op basis van de maximale winst. Vervolgens zal het geselecteerde classificatiemodel gebruikt worden om berekeningen en daarmee een vergelijking te maken met:

1. De huidige werkwijze waar alle prospects gebeld worden.
2. De voorgestelde werkzijze waar het classificatiemodel een voorselectie aan prospects maakt.

Deze vergelijking zal Blackrock (naast andere overwegingen) het diepere inzicht geven en uiteindelijk dienen om een gewogen besluit te maken met betrekking tot de overname van de Portugese bank.

# 2. Methodology

Dit onderzoek kan uitgevoerd worden met elke predictive data analysis tool setup, maar wij hebben onder andere de volgende systeemeisen gebruikt:

- Python 3.11
- PDM
- Jupyter
- Pandas
- Numpy
- Scikit-Learn

De beschrijving in de [repository] voor dit onderzoek geeft aan, hoe je dit onderzoek moet opzetten. Dit zodat het onderzoek zelf en de resultaten gevalideerd kunnen worden.

## 2.1. Dataset

De dataset voor het onderzoek laat zich als volgt beschrijven:

- tijdsreeks data
- 41,000+ instances
- 20 features
  - 10 numeriek
  - 10 categoriaal
- Target met binair karakter: yes/no
- Geen missende waarden

De data is verzameld in de periode mei 2008 t/m november 2010.

## 2.2. Data Cleaning

Afgezien dat er geen waarden missen in de dataset, is er wel een andere taak qua data cleaning vereist. De beschrijving van de dataset geeft aan dat de feature *duration* verwijderd moet worden. Dit om opmerkelijke voorsellingen te voorkomen.

Naar onze mening heeft dat te maken met mogelijke data leakage op het moment van trainen van een voorspellingsmodel. Dit met de gedachte dat een voorspelling van een gesprek vooraf niet gemaakt kan worden, als de *duration* van datzelde gesprek nog vastgelegd moet worden.

Deze *duration* feature moet daarom verwijderd worden.

## 2.3. Feature Engineering

De dataset is opgebouwd als een tijdsreeks. Echter mist er een concrete timestamp in de data. Door de periodebepaling van de databeschrijving te gebruiken - samen met de waarden in de *month* feature, kan de het jaar per instance bepaald worden. Deze nieuw aangemaakte *year* feature zal meerdere inzichten qua data analyse mogelijk maken.

## 2.4. Train Test Split



## 2.5. Preprocessing

Een aantal features in de dataset zijn kwalitatieve waarden en moeten omgevormd worden naar kwantitatieve waarden om als input voor een model te kunnen dienen. De features in kwestie zijn:

- job
- marital
- education
- contact
- poutcome

## 2.6. Model Shortlist

! Beschrijf alle modellen en waarom ze gekozen zijn:

- Neural Networks: als referentiepunt van de originele studie
- Random Forest: simpeler van aard, maar performance van neurale netwerken kan bijhouden
- AdaBoost: een iteratie op random forest met als doel om zwak ingeschatte resultaten te boosten
- XGBoost: gebaseerd op generative model, waardoor sampling mogelijk is

# 3. Exploratory Data Analysis

Om lekkage van data te voorkomen is de feature `duration` niet meegenomen in het train proces. Dit omdat de waarde hiervan niet gebruikt kan worden - wanneer er een voorspelling gedaan wordt, want de eindgebruiker (call agent) weet pas na afloop hoe lang het gesprek heeft geduurd.

# 4. Results

## 4.1. Metrics

- ROC
- Calibration
- Recall

| Metric | AdaBoost | Neural Net | Random Forest |
| ------ | -------- | ---------- | ------------- |
| AUC    | 0.7465   | 0.7626     | 0.7739        |
| ALIFT  | 0.2465   | 0.2626     | 0.2739        |

## 4.2. Model Evaluation

! Beschrijf alle evaluaties en waarom ze gekozen zijn:

- Confusion matrix
- Probalility calibration
- Cost vs threshold analysis

## 4.3. Model Selection

## 4.4. ROI Calculations

 

# 5. Discussion

# 6. Conclusion

# References