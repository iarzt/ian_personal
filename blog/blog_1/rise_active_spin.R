library(ggpmisc)
sm <- read.csv('completed.csv')
as <- read.csv('active_spin_plus_la.csv')
as <- na.omit(as)
aw <- read.csv('active_whiff.csv')

active_spin_movement <- ggplot(sm, aes(x = active_spin_fastball, y = rise)) + 
  geom_point(aes(colour = rise)) + 
  ggtitle("Fastball Rise vs. Active Spin") + 
  xlab(label = 'Fastball Active Spin') + 
  ylab(label = 'Rise') + 
  labs(subtitle = 'Evaluating the Effects of Active Spin on Fastball Rise', colour = 'Rise') + 
  theme(plot.title = element_text(hjust = 0.5, size = 14), plot.subtitle = element_text(hjust = 0.5, size = 10)) +
  geom_smooth(method = "lm", se = FALSE, colour = 'red', formula = y ~ x) + 
  stat_poly_eq(formula = y ~ x, aes(label = paste(..rr.label..)), parse = TRUE)

active_spin_movement

as$active_spin_fastballe=as.numeric(levels(as$active_spin_fastball))[as$active_spin_fastball]

formula <- y ~ x
as_launch <- ggplot(as, aes(x = active_spin_fastball, y = LA)) + 
  geom_point() + 
  ggtitle("Launch Angle vs. Active Spin") + 
  xlab(label = 'Fastball Active Spin') + 
  ylab(label = 'Launch Angle') + 
  labs(subtitle = 'Evaluating the Effects of Active Spin on Launch Angle (Four Seam Only)') + 
  ylim(0, 40) + 
  scale_x_continuous(name = "Fastball Active Spin", limits=c(25, 100)) + 
  theme(plot.title = element_text(hjust = 0.5, size = 14), plot.subtitle = element_text(hjust = 0.5, size = 10)) +
  geom_smooth(method = "lm", se = FALSE, colour = 'red', formula = y ~ x) + 
  stat_poly_eq(formula = y ~ x, aes(label = paste(..rr.label..)), parse = TRUE)

as_launch

formula <- y ~ x
as_whiff <- ggplot(aw, aes(x = active_spin_fastball, y = aw$Whiff..)) + 
  geom_point() + 
  ggtitle("Whiff % vs. Active Spin") + 
  xlab(label = 'Fastball Active Spin') + 
  ylab(label = 'Whiff %') + 
  labs(subtitle = 'Evaluating the Effects of Active Spin on Whiff Rate (Four Seam Only)') + 
  ylim(0, 40) + 
  scale_x_continuous(name = "Fastball Active Spin", limits=c(60, 100)) + 
  theme(plot.title = element_text(hjust = 0.5, size = 14), plot.subtitle = element_text(hjust = 0.5, size = 10)) +
  geom_smooth(method = "lm", se = FALSE, colour = 'red', formula = y ~ x) + 
  stat_poly_eq(formula = y ~ x, aes(label = paste(..rr.label..)), parse = TRUE)

as_whiff

velo_whiff <- ggplot(aw, aes(x = MPH, y = aw$Whiff..)) + 
  geom_point(aes(colour = MPH)) + 
  ggtitle("Whiff % vs. Average Fastball Velocity") + 
  xlab(label = 'Fastball Velocity') + ylab(label = 'Whiff %') + 
  labs(subtitle = 'Evaluating the Effects of Velocity on Whiff Rate (Four Seam Only)') + 
  ylim(0, 40) + 
  theme(plot.title = element_text(hjust = 0.5, size = 14), plot.subtitle = element_text(hjust = 0.5, size = 10)) +
  geom_smooth(method = "lm", se = FALSE, colour = 'red', formula = y ~ x) + 
  stat_poly_eq(formula = y ~ x, aes(label = paste(..rr.label..)), parse = TRUE)
velo_whiff
