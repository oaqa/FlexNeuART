#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage <mehod row> <pct, e.g., 1,5 etc> <num round digit>");
}
run    <- args[1] 
pct    <- as.numeric(args[2])
digit  <- as.numeric(args[3])

dt <- as.numeric(read.table(run   ))

m <- mean(dt)
d <- qnorm((1-pct/200.0))
print(paste("sigma coeff=",d, sep=" "))
n <- length(dt)
s <- sd(dt)
s_norm <- s/sqrt(n)
print(run)
print(paste("sd=", s, " normalized sd=", s_norm, sep=""))
print(paste("[ ", round(m - d*s_norm,digit), ",", round(m, digit), ",", round(m + d*s_norm,digit), " ]", sep=""))
print(paste(round(m, digit), '$\\pm$', round(d*s_norm,digit), sep=""))

print("----------------------------")


