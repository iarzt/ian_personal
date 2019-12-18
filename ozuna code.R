my.formula <- y~x
ozunax <- ggplot(ozuna_xwOBA, aes(XWOBA, Barrel)) + geom_smooth(method = "lm", color = "red", formula = my.formula) + stat_poly_eq(formula = my.formula, 
                                                                                                                                   aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
                                                                                                                                   parse = TRUE) + geom_point(size = 3) + ggtitle("Marcell Ozuna Predictive xwOBA by Barrel %") + xlab('xwOBA') + ylab('Barrel %') + xlim(.300, .400)
ozunax

                                                                                                                                                                                                                                                                                                                        

