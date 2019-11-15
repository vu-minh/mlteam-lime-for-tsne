##############################################
#### Translate the output of BIR into csv ####
##############################################

BIR.out.path <- "BIR/Results/"
out.path <- "results/"

load(paste0(BIR.out.path, "result.RData"))

write.csv(res$W, paste0(out.path, "BIR_W.csv"))
write.csv(res$R_squared, paste0(out.path, "BIR_R2.csv"))
