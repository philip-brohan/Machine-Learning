#!/usr/bin/Rscript

library(neuralnet)
library(GSDF.TWCR)
library(lubridate)
library(getopt)

spec<-matrix(c(
  'algorithm',    'a', 2, "character",
  'act.fct',      'f', 2, "character",
  'linear.output','l', 2, "logical",
  'hidden',       'h', 1, "integer",
  'threshold',    't', 2, "double"
), byrow=TRUE, ncol=4)
opt = getopt(spec)
if ( is.null(opt$algorithm) )     { opt$algorithm='rprop+' }
if ( is.null(opt$act.fct) )       { opt$act.fct='logistic' }
if ( is.null(opt$linear.output) ) { opt$linear.output=TRUE }
if ( is.null(opt$threshold) )     { opt$threshold=0.01 }
if ( is.null(opt$hidden) )        { stop('No hidden layer specified') }

# load the data to be modelled
ukw<-readRDS('ukw.rds')

nn<-neuralnet(at~slp+u+v+c,data=ukw,hidden=opt$hidden,
               algorithm=opt$algorithm,
               act.fct=opt$act.fct,
               linear.output=opt$linear.output,
               threshold=opt$threshold)

# Mark the output
op.string<-sprintf("nn_%s_%s_%s_%5.3f_%02d.rds",
                   opt$algorithm,opt$act.fct,
                   as.character(opt$linear.output),
                   opt$threshold,opt$hidden)

saveRDS(nn,file=op.string)
        
nn
