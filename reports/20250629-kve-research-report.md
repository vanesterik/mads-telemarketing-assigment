---
date: 29th of June 2025
title: Predicting Telemarketing Call Duration for Banking Product Subscriptions
subtitle: Predictive Modelling (2024 P3A)
institute: HAN - Master Applied Data Science
author: |
  | Koen van Esterik
  | kd.vanesterik@student.han.nl
---

# 0. Abstract



# 1. Introduction

Voor elk commercieel bedrijf bestaat de behoefte om winst te maken, om op basis daarvan investeringen te doen, groei te realiseren en uiteindelijk bestaanszekerheid te garanderen. Dit geldt zeker voor investeringsmaatschappijen die opereren op het gebied van de aanschaf en verkoop van bedrijven - waar niet alleen het realiseren van winst van belang is, maar het maximaliseren van winst.

De focus van dit onderzoeksrapport is een investeringsmaatschappij die een Portugese bank heeft aangeschaft en op zoek is naar manieren om winstmaximalisatie te realiseren.

Het toeval wil dat de Portugese bank in kwestie eerder al een onderzoek heeft uitgevoerd naar mogelijke winstmaximalisatie omtrent telemarkering campagnes. De campagnes bestaan uit telemarketeers die uitbellen naar klanten om een termijn deposito af te nemen. In veel gevallen waren meerdere contactmomenten nodig om vast te stellen of de klant het product zou afsluiten.

Het eerdere uitgevoerde onderzoek was gebasseerd op het voorspellen of een klant in een uitgaand telefoongesprek een termijn deposito zal afsluiten (ja/nee). Dit om de vereisten in te lossen om beter te presteren dan willekeurig uit te bellen.

Het resultaat hiervan dient als input voor dit onderzoeksrapport. Dit omdat de investeringsmaatschappij ook geïnteresseerd is in mogelijke winstmaximalisatie. Een winstmaximalisatie met betrekking tot het reduceren van willekeurige besluitvorming op gebied van telemarketing campagnes. 

**Door een herziening toe te passen van eerdere gebruikte en nieuwe voorspellingsmodellen en deze te onderwerpen aan verschillende evaluatie technieken waaronder `cost based threshold analysis`.**

! Beschrijf de reden dat het voor de investeringsmaatschappij interessant kan zijn om hetzelfde onderzoek nog een keer te doen, maar ook vernieuwde modellen te proberen - die mogelijk een beter resultaat kunnen leveren.

Beschrijf cost ratio en waarom deze gekozen is!

Dit onderzoek laat zich omschrijven als:

****> Improve <problem-statement>
> by redesigning <solution>
> that satisfies <requirements>
> in order to reach <stakeholder-goals>

# 2. Methodology

## 2.1. Dataset

Om lekkage van data te voorkomen is de feature `duration` niet meegenomen in het train proces. Dit omdat de waarde hiervan niet gebruikt kan worden - wanneer er een voorspelling gedaan wordt, want de eindgebruiker (call agent) weet pas na afloop hoe lang het gesprek heeft geduurd.

## 2.2. Preprocess

Een aantal features in de dataset zijn kwalitatieve waarden en moeten omgevormd worden naar kwantitatieve waarden om als input voor een model te kunnen dienen. De features in kwestie zijn:

- job
- marital
- education
- contact
- poutcome

...


## 2.3. Models

! Beschrijf alle modellen en waarom ze gekozen zijn:

- Neural Networks: als referentiepunt van de originele studie
- Random Forest: simpeler van aard, maar performance van neurale netwerken kan bijhouden
- AdaBoost: een iteratie op random forest met als doel om zwak ingeschatte resultaten te boosten
- XGBoost: gebaseerd op generative model, waardoor sampling mogelijk is

## 2.4. Evaluations

! Beschrijf alle evaluaties en waarom ze gekozen zijn:

- Confusion matrix
- Probalility calibration
- Cost vs threshold analysis

# 3. Results

| Metric | AdaBoost | Neural Net | Random Forest |
| ------ | -------- | ---------- | ------------- |
| AUC    | 0.7465   | 0.7626     | 0.7739        |
| ALIFT  | 0.2465   | 0.2626     | 0.2739        |



# 4. Discussion

# 5. Conclusion

# References