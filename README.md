## Analys av bilförsäljning baserat på drivmedelspriser

En ansats att hitta kopplingar mellan prisläget på bensin, diesel, etanol samt elpriser och fördelningen nyregistrerade bilar avseende drivmedelstyp.

De dataset som nyttjats återfinns under mappen "dataset" där även de helt orörda orginalunderlagen har fått en egen undermapp.

I analyze_data.ipynb tvättas datan för att till slut landa i en dataframe som sparas ut till filen complete_df.pkl. Denna används sedan som utgångspunkt i train_and_predict.py där ett antal modeller testas i jakt på ovan nämnda samband.

Dokumentet "reflektioner.docx" innehåller mina tankar och reflektioner.

