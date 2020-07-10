SELECT active_spin_fastball, pitch_movement.full_name, rise, pitcher_break_z, percent_rank_diff_z 
FROM `active-spin`
RIGHT JOIN pitch_movement ON pitch_movement.full_name = `active-spin`.full_name
