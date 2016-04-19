library(neuralnet)
library(GSDF.TWCR)
library(lubridate)

# Get the TWCR data - want UK region, every 6 hours
#  for a year, on the rainfall grid.

date.c<-ymd('2014-01-01')

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
  mean$data[]<-(mean$data-normal$data)/(sd$data*6)+0.5
  mean$data[]<-pmax(0,mean$data)
  mean$data[]<-pmin(1,mean$data)
  return(mean)
}

# Use the air.2m grid, trimmed to the UK region
grid<-trim(centre.on.greenwich(
         TWCR.get.slice.at.hour('air.2m',2014,1,1,0,version='3.5.1')))
  
# Get the variables of interest - need normals and sds for standardisation

prmsl<-fill.var('prmsl',grid,365,'mean')
prmsl.n<-fill.var('prmsl',grid,365,'normal')
prmsl.sd<-fill.var('prmsl',grid,365,'standard.deviation')
prmsl.std<-standardise(prmsl,prmsl.n,prmsl.sd)

air.2m<-fill.var('air.2m',grid,365,'mean')
air.2m.n<-fill.var('air.2m',grid,365,'normal')
air.2m.sd<-fill.var('air.2m',grid,365,'standard.deviation')
air.2m.std<-standardise(air.2m,air.2m.n,air.2m.sd)

uwnd.10m<-fill.var('uwnd.10m',grid,365,'mean')
uwnd.10m.n<-fill.var('uwnd.10m',grid,365,'normal')
uwnd.10m.sd<-fill.var('uwnd.10m',grid,365,'standard.deviation')
uwnd.10m.std<-standardise(uwnd.10m,uwnd.10m.n,uwnd.10m.sd)

vwnd.10m<-fill.var('vwnd.10m',grid,365,'mean')
vwnd.10m.n<-fill.var('vwnd.10m',grid,365,'normal')
vwnd.10m.sd<-fill.var('vwnd.10m',grid,365,'standard.deviation')
vwnd.10m.std<-standardise(vwnd.10m,vwnd.10m.n,vwnd.10m.sd)

# Pack the standardised data into a data frame and fit the model
ukw<-data.frame(slp=as.vector(prmsl.std$data),
                at=as.vector(air.2m.std$data),
		u=as.vector(uwnd.10m.std$data),
		v=as.vector(vwnd.10m.std$data))

nn<-neuralnet(at~slp+u+v,data=ukw,hidden=2)

plot(hexbin(as.vector(air.2m.std$data),as.vector(nn$net.result[[1]])))

fitted<-air.2m
fitted$data[]<-as.vector(nn$net.result[[1]])

compare.fit<-function(n,...) {

   f.in<-GSDF.reduce.1d(air.2m.std,'time',function(x){return(x[n])})
   f.out<-GSDF.reduce.1d(fitted,'time',function(x){return(x[n])})
   GSDF.pplot.2d(f.in,f.out,...)

}
