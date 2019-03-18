
import numpy as np

#function to find first differences in an vector

def time_dif(x):
    t=[]
    for i in range(len(x)-1):
        d=x[i+1]-x[i]
        t.append(d)
    return np.asarray(t)

#function to find first differences in an vector, keeping first value (used for cumulative
#percentages)
def time_dif1(x):
    X=[x[0]]
    for i in range(len(x)-1):
        d=x[i+1]-x[i]
        X.append(d)
    X=np.asarray(X,float)
    return X

#finds rate of change of time series
def rate_of_change(x):
    rates_vector=[]
    for i in range(len(x)-1):
        r=x[i+1]/x[i]
        rates_vector.append(r)
    rates_vector=np.asarray(rates_vector,float)
    return rates_vector

#two functions that finds moving average of length N
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_mean1(x,N):
    r=np.convolve(x, np.ones((N,))/N, mode='valid')
    return r

#Autocorrelation coefficient with lag time k

def auto_cor(x,k):
    x_bar=np.mean(x)
    N=len(x)
    z1=0
    z2=0
    for i in range(N):
        z1=z1+(x[i]-x_bar)**2

    for i in range(N-k):
        z2=z2+(x[i]-x_bar)*(x[i+k]-x_bar)

    r_k=z2/z1

    return r_k

#Autocorrelations for lags k=0,..,N/4

def AUT(x):
    N=len(x)
    AUT=[]
    for k in range(int(N/4)+1):
        AUT.append(auto_cor(x,k))

    return AUT

#Autocorrelations for all lags k=0,...,N-1

def AUT1(x):
    N=len(x)
    AUT=[]
    for k in range(N-1):
        AUT.append(auto_cor(x,k))

    return AUT

#Finds emperical distribution of data, forms step function

def ecdf(data):
    raw_data = np.array(data)
    cdfx = np.sort(raw_data)

    x_values = np.arange(min(cdfx),max(cdfx)+1,1)

    size_data = np.size(data)

    y_values = []
    for i in x_values:
        temp = raw_data[raw_data <= i]
        value = temp.size / size_data
        y_values.append(value)

    return np.array(x_values), np.array(y_values)

#Finds emperical distribution of data, getting rid of duplicate values
def ecdf_points(data):
    L=len(data)
    data_array=[]
    for i in range(L):
        data_array.append(data[i])
    np.asarray(data_array,float)
    x=np.sort(data_array)
    y=np.arange(0,1,1/L)
    X=[x[0]]
    Y=[y[0]]
    for i in range(0,L-1):
        if(x[i]!=x[i+1]):
            X.append(x[i+1])
            Y.append(y[i+1])
    X=np.asarray(X,float)
    Y=np.asarray(Y,float)
    return X,Y

#exponential function
def expon(x,a,b):
    d=b*np.exp(-a*x)
    return d
#exponential tail distribution
def expon1(x,a):
    d=np.zeros(len(x))
    for i in range(len(x)):
        d[i]=np.exp(-a*x[i])
    return d

#gamma type function
def lin_exp(x,a,b,c):
    d=np.zeros(len(x))
    for i in range(len(x)):
        d[i]=c*x[i]**b*np.exp(-a*x[i])
    return d

#pareto tail funtion
def pareto(x,a,b):
    n=len(x)
    d=np.ones(n)

    for i in range(n):
        if(x[i]>a):
            d[i]=(a/x[i])**b
    return d

#parteo tail function
def pareto1(x,a,b):
    d=np.zeros(len(x))
    for i in range(len(x)):
        d[i]=(a/x[i])**b
    return d

#basic stats function
def basic_stats(x):
    a=np.min(x)
    b=np.max(x)
    c=np.mean(x)
    d=np.median(x)

    return a,b,c,d
