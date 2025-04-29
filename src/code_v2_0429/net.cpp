#include "net.hpp"


//full connecter   is  wrong

//init full connecter    can be replaced by SXLAL generate matrix?
void Net::initNet(Mat _layer_neuron_num)
{
	layer_neuron_num = _layer_neuron_num;
	layers.resize(layer_neuron_num.size());
	// generate each layer
	static int i = 0;
	for (auto& matrix : layers)
	{

		matrix.resize(layer_neuron_num[i][0]);
		for (auto& row : matrix)
		{
			row.resize(layer_neuron_num[i][1]);
		}
		i = i + 1;
	}

	cout << "Generate layers,success! --------initNET Stage1 finished!";

	weights.resize(layers.shape()-1);
	bias.resize(layers.shape()-1);
	for (static int j = 0;j < (layer.size() - 1);++j)
	{
		weights[j].resize(layer_neuron_num[j+1][0]);
		bias[j].resize(layer_neuron_num[j+1][0]);
		for (static int k = 0;k < layer_neuron_num[j];++k)
		{
			weights[j][k].resize(layer_neuron_num[j+1][1]);
			bias[j][k].resize(layer_neuron_num[j+1][1]);
		}
	}

	cout << "Generate weights and bias,success! --------initNET Stage2 finished!";

}




//random init  full connnecter weight  using SXLAL
//M means row,N means col,to correspond with SXLAL
//example: auto randMAT = initWeight<CF,3,4>(M=3,N=4)
template<class T,int M,int N>
Mat Net::initweight_and_bias()
{
	Float_Point_Algebra fpa;
	T matrix[M][N];
	fpa.RND_MAT<T,M,N>(MATRIX);

	Mat result;
	for(int i;i < M;++i)
	{
		result.emplace_back(matrix[i],matrix[i]+N);
	}
	return result;
}




//init conv_kernel
void Net::initConvkernel(Convk kernel,int kernel_size,int i_ch,int o_ch)
{
	kernel.resize(o_ch);
	for(int i = 0;i < o_ch;i++)
	{
		kernel[i].resize(i_ch);
		for(int j = 0;j < i_ch;j++)
		{
			kernel[i][j].resize(kernel_size);
			for(int k = 0;k < kernel_size;k++)
			{
				kernel[i][j][k].resize(kernel_size);
			}

		}
	}
	Float_Point_Algebra fpa;

	for(int a = 0;a < o_ch;a++)
	{
		for(int b = 0;b < i_ch;b++)
		{
			fpa.RND_MAT_NT(kernel[a][b]);
		}
	}

}




















