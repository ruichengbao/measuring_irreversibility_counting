#include <cmath>
#include <cstddef>
#include <cstring>
extern "C" int pair_histograms_cpp(const double* y, long long T, int N, int lagsteps, int S, double L, double* out){
  if(T<=lagsteps || lagsteps<1 || N<1 || S<2) return -1;
  std::memset(out,0,sizeof(double)*2*(size_t)S*S);
  long long counts[2]={0,0}; const double scale=S/L;
  for(long long t=0;t+lagsteps<T;t++){
    const int p=t%2; counts[p]++;
    double* H=out+(size_t)p*S*S;
    const double* a=y+(size_t)t*N*2;
    const double* b=y+(size_t)(t+lagsteps)*N*2;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
      double gx=(b[2*j]-a[2*i])*scale;
      double gy=(b[2*j+1]-a[2*i+1])*scale;
      gx-=S*std::floor(gx/S);gy-=S*std::floor(gy/S);
      int ix=(int)gx,iy=(int)gy;
      if(ix==S)ix=0;if(iy==S)iy=0;
      double fx=gx-ix,fy=gy-iy;
      int jx=(ix+1)%S,jy=(iy+1)%S;
      H[(size_t)ix*S+iy]+=(1-fx)*(1-fy);
      H[(size_t)jx*S+iy]+=fx*(1-fy);
      H[(size_t)ix*S+jy]+=(1-fx)*fy;
      H[(size_t)jx*S+jy]+=fx*fy;
    }
  }
  for(int p=0;p<2;p++){
    if(!counts[p])return -2;
    double den=1.0/((double)counts[p]*N*N);
    for(size_t s=0;s<(size_t)S*S;s++)out[(size_t)p*S*S+s]*=den;
  }
  return 0;
}

extern "C" int pair_scores_cpp(const double* y, long long T, int N, int lagsteps, int S, double L, const double* J, double* out){
  if(T<=lagsteps || lagsteps<1 || N<1 || S<2) return -1;
  const double scale=S/L, invpairs=1.0/((double)N*N);
  for(long long t=0;t+lagsteps<T;t++){
    const double* a=y+(size_t)t*N*2;
    const double* b=y+(size_t)(t+lagsteps)*N*2;
    double sum=0.0;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
      double gx=(b[2*j]-a[2*i])*scale;
      double gy=(b[2*j+1]-a[2*i+1])*scale;
      gx-=S*std::floor(gx/S);gy-=S*std::floor(gy/S);
      int ix=(int)gx,iy=(int)gy;
      if(ix==S)ix=0;if(iy==S)iy=0;
      double fx=gx-ix,fy=gy-iy;
      int jx=(ix+1)%S,jy=(iy+1)%S;
      sum+=(1-fx)*(1-fy)*J[(size_t)ix*S+iy]
         +fx*(1-fy)*J[(size_t)jx*S+iy]
         +(1-fx)*fy*J[(size_t)ix*S+jy]
         +fx*fy*J[(size_t)jx*S+jy];
    }
    out[t]=sum*invpairs;
  }
  return 0;
}
