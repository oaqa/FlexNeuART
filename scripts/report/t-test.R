#!/usr/bin/env Rscript



args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage <mehod #1 row> <method #2 row> <# for Bonferroni correction>");
}
method1 <- args[1] 
method2 <- args[2] 
N <- as.numeric(args[3])

dt1 <- as.numeric(read.table(method1))
dt2 <- as.numeric(read.table(method2))

if (length(dt1) != length(dt2)) {
  stop("The number of columns/rows don't match")
}


print(paste(mean(dt1), " -> ", method1))
print(paste(mean(dt2), " -> ", method2))
t_res <- t.test(dt1,dt2, paired=TRUE)
pval <- t_res[['p.value']]
print(paste(pval, pval * N))

print("----------------------------")


