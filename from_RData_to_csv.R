##############################################
#### Translate the output of BIR into csv ####
##############################################

args <- commandArgs(TRUE)

if (length(args) < 2) {
  BIR.out.path <- "BIR/Results/result.RData"
  out.path <- "results/"
} else if (length(args) == 2) {
  BIR.out.path <- args[1]
  out.path <- args[2]
}

load(BIR.out.path)

write.csv(res$W, paste0(out.path, "BIR_W.csv"))
write.csv(res$R_squared, paste0(out.path, "BIR_R2.csv"))
