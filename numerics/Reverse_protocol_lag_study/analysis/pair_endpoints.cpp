#include <cmath>
#include <cstring>
#include <cstddef>
extern "C" int endpoint_histogram(const double* a,const double* b,int B,int N,int S,double L,double* out){
 if(B<1||N<1||S<2)return -1;
 std::memset(out,0,sizeof(double)*(size_t)S*S);
 const double scale=S/L;
 for(int t=0;t<B;++t)for(int i=0;i<N;++i)for(int j=0;j<N;++j){
  double gx=(b[((size_t)t*N+j)*2]-a[((size_t)t*N+i)*2])*scale;
  double gy=(b[((size_t)t*N+j)*2+1]-a[((size_t)t*N+i)*2+1])*scale;
  gx-=S*std::floor(gx/S);gy-=S*std::floor(gy/S);
  int ix=(int)gx,iy=(int)gy;if(ix==S)ix=0;if(iy==S)iy=0;
  double fx=gx-ix,fy=gy-iy;int jx=(ix+1)%S,jy=(iy+1)%S;
  out[(size_t)ix*S+iy]+=(1-fx)*(1-fy);
  out[(size_t)jx*S+iy]+=fx*(1-fy);
  out[(size_t)ix*S+jy]+=(1-fx)*fy;
  out[(size_t)jx*S+jy]+=fx*fy;
 }
 double norm=1./(double(B)*N*N);
 for(size_t s=0;s<(size_t)S*S;++s)out[s]*=norm;
 return 0;
}
