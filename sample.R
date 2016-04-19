library(neuralnet)

nn<-neuralnet(case~age+parity+induced+spontaneous,
              data=infert,hidden=2,err.fct='ce',
              linear.output=FALSE)
