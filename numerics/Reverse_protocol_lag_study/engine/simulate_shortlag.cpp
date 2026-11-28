#include "../../particle_species.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// The continuous model is overdamped Brownian dynamics on a square torus.
// A sweep of one-particle MALA updates integrates the relative generator.
// The midpoint field is included in the proposal and the Hastings ratio,
// enforcing local detailed balance including the external work increment.
// The center of friction is separated analytically. This provides its exact
// independent drift and diffusion and removes translational sampling noise.
struct V {double x=0,y=0;};
struct Eval {double u=0,fx=0,fy=0;};
constexpr double L=5, spring=5, r0=.001, cutoff=2, twopi=6.2831853071795864769;
inline double mi(double x){return x-L*std::nearbyint(x/L);}
struct Sim {
 int N; double dt,E,Gamma=0,Q=0,B=0,time=0; bool reverse_protocol=false; double protocol_endpoint=0; std::vector<V> x;
 std::vector<double> sig,D,q,sqrtd,weight; std::mt19937_64 rng;
 std::normal_distribution<double> normal{0,1}; std::uniform_real_distribution<double> uniform{0,1};
 uint64_t attempts=0,accepts=0; double maxradius=0; bool random_order=true;
 std::vector<Eval> cache,temporary;uint64_t sweeps=0;double cache_error=0;
 Sim(int n,double field,double step,uint64_t disorder,uint64_t noise):N(n),dt(step),E(field),x(n),sig(n),D(n),q(n),sqrtd(n),weight(n),rng(noise){
  std::mt19937_64 drng(disorder);
  for(int i=0;i<N;++i){ sig[i]=.01*(species_index(drng)+1); D[i]=.01/sig[i];q[i]=.9+.1*species_index(drng); Gamma+=1/D[i];Q+=q[i];B+=D[i]*q[i]*q[i]; sqrtd[i]=sqrt(2*D[i]*dt); }
  // Dilute square cluster with no WCA overlaps, recentered below.
  int side=ceil(sqrt(N));
  for(int i=0;i<N;++i){x[i]={.07*(i%side-.5*(side-1)),.07*(i/side-.5*(side-1))};weight[i]=1/D[i]/Gamma;}
  recenter();cache.resize(N*N);temporary.resize(N);rebuild_cache();
 }
 void recenter(){double rx=0,ry=0;for(int i=0;i<N;++i){rx+=weight[i]*x[i].x;ry+=weight[i]*x[i].y;}for(auto&v:x){v.x-=rx;v.y-=ry;}}
 Eval pair(int i,int j,V p)const{
  Eval a;
  if(j!=i){
   double dx=mi(p.x-x[j].x),dy=mi(p.y-x[j].y),rsq=dx*dx+dy*dy;
   if(rsq<1e-24){a.u=1e100;return a;}
   double r=sqrt(rsq), f=0;
   if(r<cutoff){a.u+=.5*spring*((r-r0)*(r-r0)-(cutoff-r0)*(cutoff-r0));f-=spring*(1-r0/r);}
   double s=.5*(sig[i]+sig[j]);
   if(rsq<1.2599210498948732*s*s){
    double z=s*s/rsq,z6=z*z*z;
    a.u+=4*(z6*z6-z6)+1; f+=24*(2*z6*z6-z6)/rsq;
   }
   a.fx+=f*dx;a.fy+=f*dy;
  }
  return a;
 }
 void rebuild_cache(){for(int i=0;i<N;++i)for(int j=0;j<i;++j){Eval a=pair(i,j,x[i]);if(sweeps){const Eval&b=cache[i*N+j];cache_error=std::max(cache_error,std::max(std::abs(a.u-b.u)/(1+std::abs(a.u)),std::hypot(a.fx-b.fx,a.fy-b.fy)/(1+std::hypot(a.fx,a.fy))));}cache[i*N+j]=a;cache[j*N+i]={a.u,-a.fx,-a.fy};}}
 Eval evaluate(int i,V p)const{
  Eval a;for(int j=0;j<N;++j)if(j!=i){Eval z=pair(i,j,p);a.u+=z.u;a.fx+=z.fx;a.fy+=z.fy;}return a;
 }
 Eval evaluate_cached(int i)const{
  Eval a;for(int j=0;j<N;++j){const Eval &z=cache[i*N+j];a.u+=z.u;a.fx+=z.fx;a.fy+=z.fy;}return a;
 }
 Eval evaluate_proposal(int i,V p){
  Eval a;for(int j=0;j<N;++j){Eval z=pair(i,j,p);temporary[j]=z;a.u+=z.u;a.fx+=z.fx;a.fy+=z.fy;}return a;
 }
 double field_integral(double a,double b)const{return E*((b-a)+(sin(twopi*b)-sin(twopi*a))/twopi);}
 void drive(double integral){for(int i=0;i<N;++i)x[i].x+=(D[i]*q[i]-Q/Gamma)*integral;}
 void step(){
  double protocol_time=reverse_protocol ? protocol_endpoint-(time+.5*dt) : time+.5*dt;
  double emid=E*(1+cos(twopi*protocol_time));
  // Random direction symmetrizes the conservative sequential composition.
  bool backwards=(rng()&1);int start=backwards?N-1:0,end=backwards?-1:N,inc=backwards?-1:1;
  for(int i=start;i!=end;i+=inc){
   Eval old=evaluate_cached(i);double nx=normal(rng),ny=normal(rng);
   // Tame only the proposal drift, not the potential. The exact Metropolis
   // ratio retains the Boltzmann measure and prevents singular-drift traps.
   double ci=q[i]-Q/(Gamma*D[i]);
   double ox=D[i]*dt*(old.fx+ci*emid),oy=D[i]*dt*old.fy;
   double os=1+sqrt(ox*ox+oy*oy)/(2*sqrtd[i]);ox/=os;oy/=os;
   V trial{x[i].x+ox+sqrtd[i]*nx,x[i].y+oy+sqrtd[i]*ny};
   Eval proposed=evaluate_proposal(i,trial);
   double dx=trial.x-x[i].x,dy=trial.y-x[i].y;
   double px=D[i]*dt*(proposed.fx+ci*emid),py=D[i]*dt*proposed.fy;
   double ps=1+sqrt(px*px+py*py)/(2*sqrtd[i]);px/=ps;py/=ps;
   double rx=-dx-px,ry=-dy-py;
   double loga=old.u-proposed.u+emid*ci*dx+.5*(nx*nx+ny*ny)-(rx*rx+ry*ry)/(4*D[i]*dt);
   ++attempts;
   if(loga>=0 || log(uniform(rng))<loga){x[i]=trial;++accepts;for(int j=0;j<N;++j){const Eval&a=temporary[j];cache[i*N+j]=a;cache[j*N+i]={a.u,-a.fx,-a.fy};}}
  }
  recenter();time+=dt;if((++sweeps)%1000==0)rebuild_cache();
 }
 double polarization()const{double p=0;for(int i=0;i<N;++i)p+=q[i]*x[i].x;return p;}
 double potential()const{double u=0;for(int i=0;i<N;++i)u+=evaluate(i,x[i]).u;return .5*u;}
};


// A single reversed branch supplies every lag m/M ending at its forward endpoint.
// Physical time always increases. Only the field schedule is read in reverse.
int main(int argc,char**argv){
 if(argc<12){std::cerr<<"Usage: simulate_shortlag N E dt burn prod stride M disorder_seed noise_seed output_prefix rep\n";return 1;}
 const int N=std::stoi(argv[1]),burn=std::stoi(argv[4]),prod=std::stoi(argv[5]),stride=std::stoi(argv[6]),M=std::stoi(argv[7]),rep=std::stoi(argv[11]);
 const double E=std::stod(argv[2]),dtin=std::stod(argv[3]);
 const uint64_t disorder=std::stoull(argv[8]),noise=std::stoull(argv[9]);const std::string prefix=argv[10];
 int per=int(std::llround(1/dtin));per=M*int(std::llround(double(per)/M));const double dt=1./per;const int phase_steps=per/M;
 if(prod<2 || stride<1 || M<2){std::cerr<<"Invalid time grid";return 2;}
 Sim sim(N,E,dt,disorder,noise);
 std::ofstream fw(prefix+"_forward.bin",std::ios::binary),rw(prefix+"_reverse.bin",std::ios::binary);
 std::ofstream origins(prefix+"_origins.csv"),periods(prefix+"_periods.csv");
 origins<<"branch_id,forward_end_index,end_time,phase_index,origin_cycle,noise_seed,acceptance,cache_error\n"<<std::setprecision(16);
 periods<<"period,internal_work,internal_work_ibp,p_start,p_end,potential,acceptance\n"<<std::setprecision(16);
 std::vector<double> frames;frames.reserve((prod*M+1)*(1+2*N));
 double center_error=0,coordinate_max=0;
 auto save_forward=[&](double time){
   fw.write(reinterpret_cast<const char*>(&time),8);frames.push_back(time);
   double cx=0,cy=0;
   for(int a=0;a<N;++a){fw.write(reinterpret_cast<const char*>(&sim.x[a].x),8);fw.write(reinterpret_cast<const char*>(&sim.x[a].y),8);frames.push_back(sim.x[a].x);frames.push_back(sim.x[a].y);cx+=sim.weight[a]*sim.x[a].x;cy+=sim.weight[a]*sim.x[a].y;coordinate_max=std::max(coordinate_max,std::hypot(sim.x[a].x,sim.x[a].y));}
   center_error=std::max(center_error,std::hypot(cx,cy));
 };
 const auto wall=std::chrono::steady_clock::now();
 for(int p=-burn;p<prod;++p){
   sim.time=double(p);if(p==0)save_forward(0.);
   double work=0,ibp=0,pbegin=sim.polarization();const uint64_t at0=sim.attempts,ac0=sim.accepts;
   for(int j=0;j<per;++j){
     double t=p+j*dt;sim.time=t;const double p0=sim.polarization();sim.step();const double p1=sim.polarization();
     const double emid=E*(1+cos(twopi*(t+.5*dt)));work+=emid*(p1-p0);ibp+=twopi*E*sin(twopi*(t+.5*dt))*.5*(p0+p1)*dt;
     if(p>=0 && (j+1)%phase_steps==0)save_forward(double(p)+double(j+1)/per);
   }
   if(p>=0){const double pend=sim.polarization();periods<<p<<','<<work<<','<<ibp+2*E*(pend-pbegin)<<','<<pbegin<<','<<pend<<','<<sim.potential()<<','<<double(sim.accepts-ac0)/(sim.attempts-at0)<<'\n';}
 }
 const double forward_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();
 const size_t width=1+2*N;int branch_id=0;double reverse_cache_error=0,reverse_accept_sum=0;
 for(int cycle=1;cycle<prod;cycle+=stride)for(int phase=0;phase<M;++phase){
   const size_t end_index=size_t(cycle)*M+phase,base=end_index*width;const double end_time=frames[base];
   const uint64_t branch_seed=noise+100000000000ULL+uint64_t(branch_id+1)*1000003ULL;
   Sim rev(N,E,dt,disorder,branch_seed);rev.reverse_protocol=true;rev.protocol_endpoint=end_time;
   for(int a=0;a<N;++a)rev.x[a]={frames[base+1+2*a],frames[base+2+2*a]};rev.recenter();rev.rebuild_cache();
   for(int j=0;j<per;++j){
     rev.time=j*dt;rev.step();
     if((j+1)%phase_steps==0)for(int a=0;a<N;++a){rw.write(reinterpret_cast<const char*>(&rev.x[a].x),8);rw.write(reinterpret_cast<const char*>(&rev.x[a].y),8);coordinate_max=std::max(coordinate_max,std::hypot(rev.x[a].x,rev.x[a].y));}
   }
   const double accept=double(rev.accepts)/rev.attempts;reverse_accept_sum+=accept;reverse_cache_error=std::max(reverse_cache_error,rev.cache_error);
   origins<<branch_id<<','<<end_index<<','<<end_time<<','<<phase<<','<<cycle<<','<<branch_seed<<','<<accept<<','<<rev.cache_error<<'\n';++branch_id;
 }
 fw.close();rw.close();origins.close();periods.close();
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();
 std::ofstream meta(prefix+"_meta.json");meta<<std::setprecision(16);
 meta<<"{\n\"integrator\":\"projected_driven_tamed_MALA\",\"implementation\":\"pair_cache\",\"L\":5,\"kBT\":1,\"epsilon\":1,\"spring_k\":5,\"spring_r0\":0.001,\"spring_cutoff\":2,\"period\":1,\"N\":"<<N<<",\"E\":"<<E<<",\"rep\":"<<rep<<",\"dt\":"<<dt<<",\"M\":"<<M<<",\"burn_periods\":"<<burn<<",\"production_periods\":"<<prod<<",\"origin_stride\":"<<stride<<",\"n_frames\":"<<(prod*M+1)<<",\"n_branches\":"<<branch_id<<",\"origins_per_phase\":"<<branch_id/M<<",\"Gamma\":"<<sim.Gamma<<",\"Q\":"<<sim.Q<<",\"B\":"<<sim.B<<",\"disorder_seed\":"<<disorder<<",\"noise_seed\":"<<noise<<",\"forward_acceptance\":"<<double(sim.accepts)/sim.attempts<<",\"reverse_acceptance_mean\":"<<reverse_accept_sum/branch_id<<",\"forward_cache_error\":"<<sim.cache_error<<",\"reverse_cache_error\":"<<reverse_cache_error<<",\"maximum_center_error\":"<<center_error<<",\"maximum_relative_radius\":"<<coordinate_max<<",\"forward_seconds\":"<<forward_seconds<<",\"seconds\":"<<seconds<<",\"reverse_protocol\":\"E_R(s)=E(t_end-s)\",\"particles\":[";
 for(int a=0;a<N;++a){if(a)meta<<',';meta<<"{\"sigma\":"<<sim.sig[a]<<",\"D\":"<<sim.D[a]<<",\"q\":"<<sim.q[a]<<'}';}meta<<"]\n}\n";
 std::cout<<prefix<<" N="<<N<<" E="<<E<<" origins/phase="<<branch_id/M<<" forward="<<forward_seconds<<" seconds="<<seconds<<"\n";
}
