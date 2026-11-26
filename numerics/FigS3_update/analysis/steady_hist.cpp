#include <cmath>
#include <cstring>
#include <cstddef>
extern "C" int steady_histogram_cpp(const double* y,long long T,int N,int shift,int stride,int offset,int S,double L,double* H){
 if(T<=shift || shift<1 || stride<1 || N<1 || S<2)return -1;
 std::memset(H,0,sizeof(double)*(size_t)S*S);long long count=0;double scale=S/L;
 for(long long t=offset;t+shift<T;t+=stride){
  ++count;const double* a=y+(size_t)t*N*2;const double* b=y+(size_t)(t+shift)*N*2;
  for(int i=0;i<N;++i)for(int j=0;j<N;++j){
   double gx=(b[2*j]-a[2*i])*scale,gy=(b[2*j+1]-a[2*i+1])*scale;
   gx-=S*std::floor(gx/S);gy-=S*std::floor(gy/S);
   int ix=(int)gx,iy=(int)gy;if(ix==S)ix=0;if(iy==S)iy=0;
   double fx=gx-ix,fy=gy-iy;int jx=(ix+1)%S,jy=(iy+1)%S;
   H[(size_t)ix*S+iy]+=(1-fx)*(1-fy);H[(size_t)jx*S+iy]+=fx*(1-fy);
   H[(size_t)ix*S+jy]+=(1-fx)*fy;H[(size_t)jx*S+jy]+=fx*fy;
  }
 }
 if(!count)return -2;double den=1./((double)count*N*N);for(size_t i=0;i<(size_t)S*S;++i)H[i]*=den;
 return (int)count;
}

extern "C" int steady_scores_cpp(const double* y,long long T,int N,int shift,const long long* origins,int count,int S,double L,const double* J,double* out){
 if(T<=shift || shift<1 || N<1 || S<2)return -1;
 const double scale=S/L,inverse=1./((double)N*N);
 for(int o=0;o<count;++o){
  long long t=origins[o];if(t<0 || t+shift>=T)return -2;
  const double* a=y+(size_t)t*N*2;const double* b=y+(size_t)(t+shift)*N*2;double value=0;
  for(int i=0;i<N;++i)for(int j=0;j<N;++j){
   double gx=(b[2*j]-a[2*i])*scale,gy=(b[2*j+1]-a[2*i+1])*scale;
   gx-=S*std::floor(gx/S);gy-=S*std::floor(gy/S);
   int ix=(int)gx,iy=(int)gy;if(ix==S)ix=0;if(iy==S)iy=0;
   double fx=gx-ix,fy=gy-iy;int jx=(ix+1)%S,jy=(iy+1)%S;
   value+=(1-fx)*(1-fy)*J[(size_t)ix*S+iy]+fx*(1-fy)*J[(size_t)jx*S+iy]+(1-fx)*fy*J[(size_t)ix*S+jy]+fx*fy*J[(size_t)jx*S+jy];
  }out[o]=value*inverse;
 }return 0;
}
