# -*- coding: utf-8 -*-
"""
VIPLab Image Libraries for class
ABE485/585
Armando Barreto
Spring 2018
"""
import numpy as np
import time


def version():
    print("viplab v4 04/23/2018")


def startTime():
    return time.time()

def endTime(start,label=''):
    end=time.time()
    l=end-start
    if(l<60):
        print(label,round(l,2)," seconds")
    else:
        mi=l//60
        s=(l/60.0 - mi) * 60
        print(label,round(mi,2)," min ",s," sec" )

#custom function to prepare for histogram
#provide bins as number or list of bins
# or provide the range for autobins    
def histo_prepare(data,bins=50,range=0):
    
    if type(range) is list:
        #range is provided as [min,max]
        bins=np.linspace(range[0],range[1],num=bins)
        #print("creating histo range...")
    
    hist,bin_edges=np.histogram(data, bins=bins)
    width = 0.8 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist,center,width


def majority_get(list):
    #return unique list, and counts for each value
    uniquelist,counts=np.unique(list,return_counts=True)
    #obtains the index of the max
    idmax=np.argmax(counts)
    value=uniquelist[idmax]
    return value


def LUT_load(fname):
    LUT=[]
    file=open(fname,"r")
    for line in file:
        line=line.split()
        if(len(line)==4 or len(line)==5):
          lutrow=[float(line[0]),float(line[1]),int(line[2]),int(line[3]),int(line[4])]    
          LUT.append(lutrow)
    
    file.close()    
    return LUT


def LUT_fromThematic(LUT):
    
    n=len(LUT)
    LUT2=[]
    for i in range(0,n):
        lr=LUT[i]
        lr=[lr[0], lr[0], lr[1], lr[2], lr[3]]
        LUT2.append(lr)
    
    return LUT2

def LUT_getcolor(LUT,nlut, value, index=0):
    #index=0
    found=False

    while index<nlut and found==False:
        lutrow=LUT[index]
        if value>=lutrow[0] and value<=lutrow[1]:
            found=True
        else:
            index=index+1
    
    
    if found==False:
       index=-1
    
    return index


def getImage_fromLUT(band,LUT,colordefault=0):
    nrows,ncols=band.shape
    
    #default color for value not found in LUT
    if(colordefault==0):
        colordefault=[191, 0, 255]
    
    if type(LUT) is str:
        lutfname=LUT
        LUT=LUT_load(lutfname)
    
    nlut=len(LUT)    
    
    #optimization technique: divide lut for scanning
    optimize= (nlut > 12)
    if optimize == True:
      #get half lut item for speed
      n25= nlut // 4
      lutrow=LUT[n25]
      vmin25=lutrow[0]
      n50= nlut // 2
      lutrow=LUT[n50]
      vmin50=lutrow[0]
    
      n75= n50+n25
      lutrow=LUT[n75]
      vmin75=lutrow[0]
    
      #print("Optimizing LUT...")
      #print("i25=",n25," vmin=",vmin25)
      #print("i50=",n50," vmin=",vmin50)
      #print("i75=",n75," vmin=",vmin75)
    #end optimize       

    index=0 #start with first color in lut
    lutrow=LUT[0]
    if(len(lutrow)==4):
       LUT=LUT_fromThematic(LUT)
       lutrow=LUT[0]
    
    vmin=LUT[0][0]
    vmax=LUT[nlut-1][1]
    
    color=[lutrow[2],lutrow[3],lutrow[4]]
    datares=np.zeros((nrows,ncols,3),np.uint8)
    for i in range(0,nrows):
        for j in range(0,ncols):
            value = band[i,j]
            if(value<lutrow[0] or value>lutrow[1] or index==-1):
                if(value>=vmin and value<=vmax):                    
                    if optimize==True:
                        if(value>=vmin50):
                            if(value>=vmin75):
                                index=n75
                            else:
                                index=n50
                        else:
                          if(value>=vmin25):
                             index=n25
                          else:    
                             index=0
                    else:
                       #no optimization
                       if(value<lutrow[0]):
                          index=0
                    
                    index=LUT_getcolor(LUT,nlut,value,index)
                else:
                   # out of the lut range
                   index=-1
                
                if(index<0):
                    color=colordefault
                else:
                    lutrow=LUT[index]
                    color=[lutrow[2],lutrow[3],lutrow[4]]
                
                
            datares[i,j,0]=color[0]
            datares[i,j,1]=color[1]
            datares[i,j,2]=color[2]   

    return datares

def LUT_getdefault(s):

    LUT=[]
    if s=='NOYES':
        LUT=[[0,255,255,255],[1,0,153,0]]
    elif s=='AEROSOL':
        LUT=[[0,223,223,223],[1,0,100,0],[2,255,128,0],[3,255,0,0]]
    elif s=='MODLAND':
        LUT=2   

    return LUT


#BandRed_AC_DOS=np.where(BandRed_AC_DOS<0,0,BandRed_AC_DOS)

def band_normalize(band, fdiv, MaxV):
    nrows,ncols=band.shape
    
    if(fdiv!=1):
      mdiv = 1 / fdiv
      band = band * mdiv
    
    MinV=band.min()    
    f=1/MaxV
    datares=MinV + band *f
    for i in range(0,nrows):
        for j in range(0,ncols):
            value = datares[i,j]
            #value=MinV+value/MaxV
            if value<0:
                datares[i,j]=0
            elif value>1:
                datares[i,j]=1
    
    return datares

def Image_getRGB(bandR,bandG,bandB, fdiv, maxV=0):    
    nrows,ncols=bandR.shape
    
    if(maxV<=0):
       MaxV=bandR.max()/fdiv
       if MaxV>0.4:
         MaxV=0.4
    else:
      MaxV=maxV/fdiv
    
    datares=np.zeros((nrows,ncols,3))
    datares[:,:,0]= band_normalize(bandR, fdiv,MaxV)
    datares[:,:,1]= band_normalize(bandG, fdiv,MaxV)
    datares[:,:,2]= band_normalize(bandB, fdiv,MaxV)
    
    return datares


def band_masking(data,datamask,masklistvalues, FILL_VALUE=-13000):
    #get size of input band
    nrows,ncols=data.shape
    
    #create empty band
    datares=np.zeros((nrows,ncols))
    for i in range(0,nrows):
        for j in range(0,ncols):
            pixelMask=datamask[i,j]
            if pixelMask in masklistvalues:
                datares[i,j]=data[i,j]
            else:
                datares[i,j]=FILL_VALUE
            
    return datares

def band_avg(band, minval, maxval):
    nrows,ncols=band.shape
    
    sumall=0
    ncount=0
    fmulti=0.0001
    for i in range(0,nrows):
        for j in range(0,ncols):
            value = band[i,j]
            if(value>=minval and value<=maxval):
                sumall=sumall+value*fmulti
                ncount=ncount+1
                
    if(ncount>0):
       average=sumall / (ncount*fmulti)
    else:
       average=-15000            
    
    return average

def band_subset(band, inirow,inicol,endrow,endcol):
    datares=band[inirow:endrow+1,inicol:endcol+1]
    return datares

def calc_NDVI(red,NIR,FILL_VI):
    NDVI=FILL_VI
    
    denominator=NIR+red
    if(denominator!=0):
        NDVI=(NIR-red) / denominator
        if(NDVI<-1 or NDVI>1):
            NDVI=FILL_VI
        else:
            NDVI=int(NDVI*10000)
    return NDVI

def calc_EVI2(red, NIR, FILL_VI):
    EVI2=FILL_VI
    
    denominator=(NIR+2.4*red+1.0)
    if(denominator!=0):
        EVI2=(2.5*(NIR-red)/denominator)
        if(EVI2<-1 or EVI2>1):
            EVI2=FILL_VI
        else:
            EVI2=int(EVI2*10000)
    
    return EVI2

def calc_EVI(red, NIR, blue, EVI2, FILL_VI):
    EVI=EVI2
    
    if(blue>0 and blue<1):
       if(NIR>red and red>blue):
           denominator=(NIR+6.0*red-7.5*blue+1.0)
           if(denominator!=0):
               EVI= 2.5*((NIR-red)/denominator)
               if(EVI<-1 or EVI>1):
                    EVI=EVI2
               else:
                   EVI=int(EVI*10000)
               
    return EVI

