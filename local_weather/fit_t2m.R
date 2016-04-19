library(neuralnet)
library(GSDF.TWCR)
library(lubridate)

# Get the TWCR data - want UK region, every 6 hours
#  for a year, on the rainfall grid.

date.c<-ymd('2014-01-01')

prate<-TWCR.get.slice.at.hour('prate',2014,1,1,0,version='3.5.1')

# switch from 0-360 to -180,180 longitudes
centre.on.greenwich<-function(field) {
    p2<-field
    lon.d<-GSDF.find.dimension(p2,'lon')
    w<-which(p2$dimensions[[lon.d]]$values>180)
    p2$dimensions[[lon.d]]$values[w]<-p2$dimensions[[lon.d]]$values[w]-360
    o<-order(p2$dimensions[[lon.d]]$values)
    p2$dimensions[[lon.d]]$values<-p2$dimensions[[lon.d]]$values[o]
    field<-GSDF.regrid.2d(field,p2)
    return(field)
}
# trim to European area
trim<-function(field,lat=c(40,70),lon=c(-25,25)) {
    f2<-field
    lon.d<-GSDF.find.dimension(f2,'lon')
    w<-which(f2$dimensions[[lon.d]]$values>=-25 &
             f2$dimensions[[lon.d]]$values<=25)
    f2$dimensions[[lon.d]]$values<-f2$dimensions[[lon.d]]$values[w]
    lat.d<-GSDF.find.dimension(f2,'lat')
    w<-which(f2$dimensions[[lat.d]]$values<=70 &
             f2$dimensions[[lat.d]]$values>=40)
    f2$dimensions[[lat.d]]$values<-f2$dimensions[[lat.d]]$values[w]
    dims<-dim(f2$data)
    dims[lat.d]<-length(f2$dimensions[[lat.d]]$values)
    dims[lon.d]<-length(f2$dimensions[[lon.d]]$values)
    f2$data<-array(dim=dims)
    field<-GSDF.regrid.2d(field,f2)
    return(field)
}

# Make the 3d structures to be filled by the weather data
fill.var<-function(var,grid,n.days,type='mean') {
   full<-grid
   full$data<-array(dim=c(length(grid$dimensions[[1]]$values),
                          length(grid$dimensions[[2]]$values),
                          n.days*4))
   full$dimensions[[3]]<-list(type='time',
                           values=rep(grid$dimensions[[3]]$values[1],366*4))
   count<-1
   for(days in seq(0,n.days-1)) {
     d.current<-date.c+days(days)
     for(hr in c(0,6,12,18)) {
        yr<-year(d.current)
        mo<-month(d.current)
        dy<-day(d.current)
        version<-'3.5.1'
        if(type=='normal' || type=='standard.deviation') version<-'3.4.1'
        field<-TWCR.get.slice.at.hour(var,yr,mo,dy,hr,
                                      version=version,type=type)
        field<-centre.on.greenwich(field)
        field<-GSDF.regrid.2d(field,grid)
        full$data[,,count]<-field$data
        full$dimensions[[3]]$values[count]<-field$dimensions[[3]]$values[1]
        count<-count+1
      }
   }
   return(full)
}

standardise<-function(mean,normal,sd) {
  mean$data[]<-(mean$data-normal$data)/(sd$data*3)
  mean$data[]<-pmax(0,mean$data)
  mean$data[]<-pmin(1,mean$data)
  return(mean)
}

# Use the precip grid, trimmed to the UK region
grid<-trim(centre.on.greenwich(
         TWCR.get.slice.at.hour('prate',2014,1,1,0,version='3.5.1')))
  
# Get the variables of interest - need normals and sds for standardisation

prmsl<-fill.var('prmsl',grid,100,'mean')
prmsl.n<-fill.var('prmsl',grid,100,'normal')
prmsl.sd<-fill.var('prmsl',grid,100,'standard.deviation')
prmsl.std<-standardise(prmsl,prmsl.n,prmls.sd)



nn<-neuralnet(case~age+parity+induced+spontaneous,
              data=infert,hidden=2,err.fct='ce',
              linear.output=FALSE)
