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

#define Vec vector<CF>
#define Mat vector<vector<CF>>
#define Tri vector<vector<vector<CF>>>

#define WEIGHT 256
#define HIGH 256

//#define LAYERS_NUM 10
//#define LAYER

using namespace std;

namespace liu
{
	class Net
	{
	public:
		Mat layer_neuron_num;  //double
		Tri layers;  //triple
		Tri weights;  //triple
		Tri bias;  //triple
		Mat layer;  //double
	public:
		//initialize net to generate params
		Net(){};
		~Net(){};

		void initNet(vector<CF> _layer_neuron_num);

		void initWeights();

		void initBias();

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
