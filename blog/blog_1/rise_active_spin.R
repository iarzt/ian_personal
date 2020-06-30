library(ggpmisc)
sm <- read.csv('completed.csv')

active_spin_movement <- ggplot(sm, aes(x = active_spin_fastball, y = rise)) + 
  geom_point(aes(colour = rise)) + ggtitle("Fastball Rise vs. Active Spin") + xlab(label = 'Fastball Active Spin') + ylab(label = 'Rise') + 
  labs(subtitle = 'Evaluating the Effects of Active Spin on Fastball Rise', colour = 'Rise') + 
  theme(plot.title = element_text(hjust = 0.5, size = 14), plot.subtitle = element_text(hjust = 0.5, size = 10)) +
  geom_smooth(method = "lm", se = FALSE, colour = 'red', formula = y ~ x) + stat_poly_eq(formula = y ~ x, 
                                                                                         aes(label = paste(..rr.label..)), 
                                                                                         parse = TRUE)
active_spin_movement

