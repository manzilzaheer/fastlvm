#ifndef FAST_RAND_H_
#define FAST_RAND_H_

#include <chrono>
#include <cmath>

/* This RNG passes TestU01 */
class xorshift128plus
{
  public:
    xorshift128plus()
    {
        s[0] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        s[1] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    ~xorshift128plus() {   }
    
    inline uint64_t rand()
    {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23; // a
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
        return (s[1] + y);
    }

    inline double rand_double()
    {
        //return ((uint32_t)(rand()&0xffffffff)) * 2.3283064e-10 ; // 5.4210108624275e-20;
        return rand() * 5.4210108624275e-20;
    }
    
    inline size_t rand_k(size_t K)
    {
        //return static_cast<unsigned>(rand() * 4.6566125e-10 * K);
        return rand()%K;
    }
    
    inline unsigned rand_k(unsigned K)
    {
        //return static_cast<unsigned>(rand() * 4.6566125e-10 * K);
        return rand()%K;
    }
    
    inline unsigned short rand_k(unsigned short K)
    {
        //return static_cast<unsigned>(rand() * 4.6566125e-10 * K);
        return rand()%K;
    }
    
    inline uint32_t rand_b(unsigned B)
    {
        return rand() & ((1U << B)-1);
    }
    
    double rand_normal()
    {
        if (cached_normal_available)
        {
            cached_normal_available = false;
            return cached_normal;
        }
        else
        {
            double u, v, s;
            do
            {
                u = 2 * rand_double() - 1; // between -1 and 1
                v = 2 * rand_double() - 1; // between -1 and 1
                s = u * u + v * v;
            }
            while (s >= 1 || s == 0);
            s = sqrt(-2 * log(s) / s);
            cached_normal = v * s;
            cached_normal_available = true;
            return u * s;
        }
    }
    
    double rand_normal(double mu, double sigma)
    {
        if (cached_normal_available)
        {
            cached_normal_available = false;
            return cached_normal * sigma + mu;
        }
        else
        {
            double u, v, s;
            do
            {
                u = 2 * rand_double() - 1; // between -1 and 1
                v = 2 * rand_double() - 1; // between -1 and 1
                s = u * u + v * v;
            }
            while (s >= 1 || s == 0);
            s = sqrt(-2 * log(s) / s);
            cached_normal = v * s;
            cached_normal_available = true;
            return u * s * sigma + mu;
        }
    }
    
    double rand_gamma(double rr)
    {
        //return new GammaDistribution(rr, 1.0).sample();
        double bb, cc;
        double xx, yy, zz;
        double result;

        if (rr <= 0.0)                  // Not well defined, set to zero and skip.
            result = 0.0;
        else if (rr == 1.0)             // Exponential
            result = -log(rand_double());
        else if (rr < 1.0)              // Johnks generator
        {
            bb = 1.0 / rr;
            cc = 1.0 / (1.0 - rr);
            do
            {
                xx = pow(rand_double(), bb);
                yy = xx + pow(rand_double(), cc);
            }
            while (yy > 1.0);
        
            result = -log(rand_double()) * xx / yy;
        }
        else                            // Marsaglia and Tsang’s Method
        { 
            bb=rr-1./3.;
            cc=1./sqrt(9.*bb);
            while(true)
            {
                xx=rand_normal();
                yy=1.+cc*xx;
                if(yy>=0)
                {
                    yy=yy*yy*yy; 
                    zz=rand_double();
                    if( (zz<=1.-.0331*(xx*xx)*(xx*xx)) || (log(zz)<=0.5*xx*xx+bb*(1.-yy+log(yy))) )
                        break;
                }
            }
            result = (bb*yy);
        }
        return result;
    }

    double rand_gamma(double rr, double ss)
    {
        return rand_gamma(rr)/ss;
    }
    
    double rand_beta(double a, double b)
    {
        double x = rand_gamma(a);
        return x/(x+rand_gamma(b));
    }
    
    private:
    // No copying allowed
    xorshift128plus(const xorshift128plus &other) =delete;
    void operator=(const xorshift128plus &other) =delete;
    
    /* seed */
    uint64_t s[2];
    bool cached_normal_available = false;
    double cached_normal;
    
};

#endif
