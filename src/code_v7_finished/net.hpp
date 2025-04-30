#ifndef NET_H
#define NET_H
#endif
#include <iostream>
#include <vector>
#include <algorithm>
#include "xfxpt_algebra.hpp"
#include <stdexcept>

#define W 4
#define I 2
#define CF ap_fixed<W,I>
#define CFL ap_fixed<(2*W),(2*I)>


#define WEIGHT 256
#define HIGH 256

using namespace std;

using Vec = vector<CF>;
using Mat = vector<vector<CF>>;
using Tri = vector<vector<vector<CF>>>;
using Convk = vector<vector<vector<vector<CF>>>>;
using Conv_config = vector<tuple<int,int,int,int,int>>;
using Lite_config = vector<tuple<int,int,int,int>>;//(i_ch,o_ch,expand_ratio,stride)


namespace liu
{
	class Net
	{
	public:

//		//to create full connecter
//		Mat layer_neuron_num;  //double
//		Tri layers;  //triple
//		Tri weights;  //triple
//		Tri bias;  //triple
//		Mat layer;  //double

		//using SXLAL
		Float_Point_Algebra fpa;


		//to create conv layers
		vector<Tri> conv_layers;
		vector<Convk> conv_kernels;
		vector<Vec> conv_biases;
		Conv_config conv_configs;//configs
		int Conv_stage_number = 0;//to record the current number of Convstage


		//to save feature pictures
		Tri input_buffer;
		Tri output_buffer;

		//to create efficient-lite3
		Lite_config lite_configs;



	public:
		//initialize net to generate params
		Net(){};
		~Net(){};

		void initNet(vector<CF> _layer_neuron_num);

		void initWeights();

		void initBias();
		//////////////////////////

		void initConvkernel(Convk kernel,int kernel_size,int i_ch,int o_ch);

		void initConvbias(Vec& bias,int o_ch);

		void add_conv_layer(int i_ch,int o_ch,int kernel_size,int stride = 1,bool padding = 0);

		Tri compute_conv_layer(const Tri& input);

		Tri residual_block(const Tri& input,const Convk& kernel1,const Mat& bias1,
						   const Convk& kernel2,const Mat& bias2);

		Tri fusion_layer(const Tri& input1,const Tri& input2);

		Tri efficient_net(const Tri& input);


		void initlitenet();

		void add_efficient_block(int i_ch,int o_ch,int expand,int stride);

		Tri forward(Tri& input);

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
