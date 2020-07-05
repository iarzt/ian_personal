SELECT Name, LA, active_spin_fastball FROM scrape
LEFT JOIN `active-spin` ON scrape.Name = full_name_code
WHERE `Pitch Type` = 'Four Seamer' AND Year = '2019';