SELECT player_name1, active_spin_fastball, EV, LA XBA, XSLG, XWOBA
FROM active_spin_examples
LEFT JOIN `active-spin` ON player_name1 = full_name
WHERE time = '2019' AND pitch_type = 'Four Seamer';