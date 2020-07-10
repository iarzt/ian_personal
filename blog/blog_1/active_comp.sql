SELECT `active-spin`.full_name, `active-spin`.active_spin_fastball, active2018.active_spin_fastball 
FROM `active-spin`
RIGHT JOIN active2018 ON `active-spin`.full_name_code = active2018.full_name
