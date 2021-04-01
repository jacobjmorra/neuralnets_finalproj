pacman::p_load(tidyverse, reshape)
raw_output = read.csv("raw_output.csv")

#Recording numerosity as a variable
for (i in 1:10) {
  starti = (i - 1)*80 + 1
  endi = i*80
  cat("\nStart:", starti, "\tEnd:", endi, "\tValue:", i)
  raw_output$numerosity[starti:endi] = i
}

#Creating dataset of ANOVA p-values
raw_output$numerosity = as.factor(raw_output$numerosity)
unit = 1:108
unit_sig = replicate(108, 1)
for (i in 1:108) {
  unit_sig[i] = aov_output = summary(aov(raw_output[,i] ~ raw_output$numerosity))[[1]][1,5]

}

# Units plus significance values
sig_units = cbind(unit, unit_sig)
cat("Units:", length(sig_units[,1]))
sig_units = sig_units[unit_sig < .01,]
sig_units[,1] = as.integer(sig_units[,1])

cat("Significant units:", length(sig_units[,1]))

# Let's find most significant units!!
cat("\nMost to least significant units:\n")
for (i in 1:10) {
  this_unit = sig_units[which.min(sig_units[,2]),]
  cat("Unit:", this_unit[1], "\tp:", this_unit[2], "\n")
  sig_units = sig_units[-which.min(sig_units[,2]),]
}

# Keeping only 10 most sig units in data
uoi = raw_output[,c(5,6,2,8,9,41,23,3,4,86,109)]
colnames(uoi) = c("5","6","2","8","9","41","23","3","4","86","numerosity")
uoi = melt(uoi, id ="numerosity" )
colnames(uoi) = c("Numerosity", "Unit", "Activation")


uoi_summary = uoi %>% group_by(Numerosity, Unit) %>% summarise(mean = mean(Activation))
uoi_summary %>% ggplot(., aes(x=Numerosity, y = mean, group = Unit, color = Unit)) + geom_point(size = 2) + geom_line(size = 1.2)




