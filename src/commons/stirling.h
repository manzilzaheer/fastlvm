#ifndef _STIRLING_H
#define	_STIRLING_H

#include <vector>
#include <cmath>
#include <mutex>

#define log_zero -10000.0

class Stirling
{
	std::mutex mtx1;
	std::vector<double*> log_stirling_num_;
	std::mutex mtx2;
	std::vector<double*> u_stirling_ratio_;
	double log_sum(double log_a, double log_b);

public:
	Stirling();
	~Stirling();
	double get_log_stirling_num(unsigned n, unsigned m);	//return the log of stirling(n,m)
	double uratio(unsigned n, unsigned m);	//return the ratio of stirling(n+1,m)/stirling(n,m)
	double vratio(unsigned n, unsigned m);	//return the ratio of stirling(n+1,m+1)/stirling(n+1,m)
	double wratio(unsigned n, unsigned m);	//return the ratio of stirling(n+1,m+1)/stirling(n,m)
};

#endif
