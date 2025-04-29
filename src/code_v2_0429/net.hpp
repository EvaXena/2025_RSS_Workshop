#ifndef NET_H
#define NET_H
#endif
#include <iostream>
#include <vector>
#include <algorithm>
#include "xfxpt_algebra.hpp"


#define W 4
#define I 2
#define CF ap_fix<W,I>


#define WEIGHT 256
#define HIGH 256

using namespace std;

using Vec = vector<CF>;
using Mat = vector<vector<CF>>;
using Tri = vector<vector<vector<CF>>>;
using Convk = vector<vector<vector<vector<CF>>>>;

namespace liu
{
	class Net
	{
	public:

		//to create full connecter
		Mat layer_neuron_num;  //double
		Tri layers;  //triple
		Tri weights;  //triple
		Tri bias;  //triple
		Mat layer;  //double


		//to create conv_layers
		vector<Tri> cov_layers;
		vector<Convk> kernels;
		Mat biases;

	public:
		//initialize net to generate params
		Net(){};
		~Net(){};

		void initNet(vector<CF> _layer_neuron_num);

		void initWeights();

		void initBias();

		void initConvkernel(Convk kernel,int kernel_size,int i_ch,int o_ch);

		void forward();

		void backward();

	protected:
		//init single layer's weight
		template<class T,int M,int N>
		Mat initweight_and_bias();
		//compute delta error
		void deltaError();
		//update weights
		void updateWeights();


	};

}
