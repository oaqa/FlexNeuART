#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage <pct, e.g., 1,5 etc>");
}
pct    <- as.numeric(args[1])

print(1-pct/200.0)
d <- qnorm((1-pct/200.0))
print(paste("sigma coeff=",d, sep=" "))

