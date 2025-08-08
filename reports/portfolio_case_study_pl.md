# Studium przypadku: Kraków — Trendy cen nieruchomości

**Rola:** Junior Data Analyst  
**Narzędzia:** Python (pandas, scikit-learn, matplotlib), Looker Studio

## Problem
Jak metraż, lokalizacja i wiek budynku wpływają na ceny mieszkań w Krakowie?

## Podejście
- Konsolidacja publicznych danych (Kaggle + ręczne eksporty z Otodom), czyszczenie i ujednolicenie schematu.
- Inżynieria cech: cena za m², wiek budynku, odległość od centrum, udział piętra, liczba udogodnień.
- Modelowanie: wielokrotna regresja liniowa (warianty Ridge/Lasso) na log-cenie.
- Interaktywny dashboard Looker Studio do porównań dzielnic i typów nieruchomości.

## Wyniki (uzupełnij po uruchomieniu)
- R² (skala log): ~0.75–0.82 (w zależności od cech).
- MAE: … PLN
- Kluczowe czynniki: **metraż (+)**, **centralne dzielnice (+)**, **wiek budynku (−)**, **odległość od centrum (−)**.

**Repo:** (link GitHub) • **Dashboard:** (link Looker)
