#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

// Physical moments of the original scalar box-counting observable.
// Box order is x-fast row-major: index=iy*K+ix.
// qmc[s,:] gives Xx/L, Xy/L, standard-normal zx, standard-normal zy.
extern "C" int box_current_moments_cpp(
    const double* y, long long T, int N, const long long* origins, int O,
    int lagsteps, int K, double L, double drift, double sigma,
    const double* qmc, long long S, int reflect_y, double* output) {
  if(N<1 || K<1 || L<=0 || sigma<0 || lagsteps<1 || S<1 || O<1) return -1;
  const int M=K*K;
  const double scale=K/L, invN2=1.0/(double(N)*N);
  std::vector<double> shifts(4*S), a(2*N), b(2*N);
  std::vector<int> ca(M),cb(M);
  auto wrap=[L](double x){return x-L*std::floor(x/L);};
  for(long long s=0;s<S;++s){
    shifts[4*s]=L*qmc[4*s];shifts[4*s+1]=L*qmc[4*s+1];
    shifts[4*s+2]=wrap(shifts[4*s]+drift+sigma*qmc[4*s+2]);
    shifts[4*s+3]=wrap(shifts[4*s+1]+sigma*qmc[4*s+3]);
  }
  for(int o=0;o<O;++o){
    const long long t=origins[o];
    if(t<0 || t+lagsteps>=T) return -2;
    for(int i=0;i<2*N;++i){a[i]=wrap(y[t*2*N+i]);b[i]=wrap(y[(t+lagsteps)*2*N+i]);}
    long double mean=0,second=0;
    for(long long s=0;s<S;++s){
      std::fill(ca.begin(),ca.end(),0);std::fill(cb.begin(),cb.end(),0);
      const double ax=shifts[4*s],ay=shifts[4*s+1],bx=shifts[4*s+2],by=shifts[4*s+3];
      for(int i=0;i<N;++i){
        double x=a[2*i]+ax,y0=a[2*i+1]+ay;
        if(x>=L)x-=L;if(y0>=L)y0-=L;
        int ix=int(x*scale),iy=int(y0*scale);
        if(ix>=K)ix=K-1;if(iy>=K)iy=K-1;
        ++ca[iy*K+ix];
        x=b[2*i]+bx;y0=b[2*i+1]+by;
        if(x>=L)x-=L;if(y0>=L)y0-=L;
        ix=int(x*scale);iy=int(y0*scale);
        if(ix>=K)ix=K-1;if(iy>=K)iy=K-1;
        ++cb[iy*K+ix];
      }
      long long xnumerator=0,ynumerator=0;int prefixrows=0;
      for(int iy=0;iy<K;++iy){
        int ar=0,br=0;for(int ix=0;ix<K;++ix){ar+=ca[iy*K+ix];br+=cb[iy*K+ix];}
        ynumerator+=(long long)br*(2*prefixrows+ar-N);prefixrows+=ar;
        int prefix=0;
        for(int ix=0;ix<K;++ix){const int m=iy*K+ix;xnumerator+=(long long)cb[m]*(2*prefix+ca[m]-ar);prefix+=ca[m];}
      }
      const double Jx=xnumerator*invN2,Jy=ynumerator*invN2;
      if(reflect_y){mean+=Jx;second+=Jx*Jx+Jy*Jy;}
      else {const double J=Jx+Jy;mean+=J;second+=J*J;}
    }
    output[2*o]=double(mean/S);output[2*o+1]=double(second/S);
  }
  return 0;
}
