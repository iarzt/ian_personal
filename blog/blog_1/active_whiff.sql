SELECT Name, active_spin_fastball, MPH, `Whiff %` FROM scrape
RIGHT JOIN `active-spin` ON full_name_code = scrape.Name
WHERE scrape.Year = '2019' AND `Pitch Type` = 'Four Seamer' AND scrape.`#` > 200;