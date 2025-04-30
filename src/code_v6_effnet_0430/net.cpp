#include "net.hpp"

//full connecter   is  wrong

//init full connecter    can be replaced by SXLAL generate matrix?
void Net::initNet(Mat _layer_neuron_num) {
	layer_neuron_num = _layer_neuron_num;
	layers.resize(layer_neuron_num.size());
	// generate each layer
	static int i = 0;
	for (auto &matrix : layers) {

		matrix.resize(layer_neuron_num[i][0]);
		for (auto &row : matrix) {
			row.resize(layer_neuron_num[i][1]);
		}
		i = i + 1;
	}

	cout << "Generate layers,success! --------initNET Stage1 finished!";

	weights.resize(layers.shape() - 1);
	bias.resize(layers.shape() - 1);
	for (static int j = 0; j < (layer.size() - 1); ++j) {
		weights[j].resize(layer_neuron_num[j + 1][0]);
		bias[j].resize(layer_neuron_num[j + 1][0]);
		for (static int k = 0; k < layer_neuron_num[j]; ++k) {
			weights[j][k].resize(layer_neuron_num[j + 1][1]);
			bias[j][k].resize(layer_neuron_num[j + 1][1]);
		}
	}

	cout
			<< "Generate weights and bias,success! --------initNET Stage2 finished!";

}

//random init  full connnecter weight  using SXLAL
//M means row,N means col,to correspond with SXLAL
//example: auto randMAT = initWeight<CF,3,4>(M=3,N=4)
template<class T, int M, int N>
Mat Net::initweight_and_bias() {
//	Float_Point_Algebra fpa;
	T matrix[M][N];
	fpa.RND_MAT<T, M, N>(MATRIX);

	Mat result;
	for (int i; i < M; ++i) {
		result.emplace_back(matrix[i], matrix[i] + N);
	}
	return result;
}

//init conv_kernel
void Net::initConvkernel(Convk &kernel, int kernel_size, int i_ch, int o_ch) {
	kernel.resize(o_ch);
	for (int i = 0; i < o_ch; i++) {
		kernel[i].resize(i_ch);
		for (int j = 0; j < i_ch; j++) {
			kernel[i][j].resize(kernel_size);
			for (int k = 0; k < kernel_size; k++) {
				kernel[i][j][k].resize(kernel_size);
			}

		}
	}

//	Float_Point_Algebra fpa;
	for (int a = 0; a < o_ch; a++) {
		for (int b = 0; b < i_ch; b++) {
			fpa.RND_MAT_NT(kernel[a][b]);
		}
	}

}

//init conv_bias
void Net::initConvbias(Vec &bias, int o_ch) {
	bias.resize(o_ch);
//	Float_Point_Algebra fpa;
	fpa.RND_VEC_NT(bias);

}

//generate a full conv_layer
void Net::add_conv_layer(int i_ch, int o_ch, int kernel_size, int stride = 1,
		bool padding = 0) {
	//save this layer's cfg
	conv_configs.emplace_back(i_ch, o_ch, kernel_size, stride, padding);
	//create this layer's kernel and save it to kernels
	Convk kernel;
	initConvkernel(kernel, kernel_size, i_ch, o_ch);
	conv_kernels.push_back(kernel);
	//create this layer's bias and save it to biases
	Vec bias;
	initConvbias(bias);
	conv_biases.push_back(bias);
}

//compute a full conv_layer
Tri Net::compute_conv_layer(const Tri &input) {
	auto& [i_ch, o_ch, kernel_size, stride, padding] =
			conv_configs[Conv_stage_number];
	auto &kernel = conv_kernels[Conv_stage_number];
	auto &bias = conv_biases[Conv_stage_number];
	Tri output;
	fpa.CONV_2D_NT(input, kernel, bias, output, padding, stride);
	Conv_stage_number = Conv_stage_number + 1;
	return output;

}

//0430
//compute a residual_block
Tri Net::residual_block(const Tri& input)
{
//	const int in_ch = input.size();
//	const int h = input[0].size();
//	const int w = input[0][0].size();
//
//	if(kernel1.empty() || kernel2.empty() || kernel1[0].size()!=in_ch || kernel2[0].size()!=kernel1.size())
//	{
//		throw std::invalid_argument("Kernel channel mismatch!");//no catch?
//	}
//
//	const int out_ch = kernel2.size();
//	if(out_ch != in_ch)
//	{
//		throw std::invalid_argument("IN/OUT channel mismatch in residual!");//no catch?
//	}

	//1st relu
	Tri relu1_out = input;
	for (auto& channel : relu1_out)
	{
		fpa.RELU_MAT(channel);
	}
	//1st conv 3x3
	Tri conv1_out = compute_conv_layer(relu1_out);
	//2nd relu
	for (auto& channel : conv1_out)
	{
		fpa.RELU_MAT(channel);
	}
	//2nd conv 3x3
	Tri conv2_out = compute_conv_layer(conv1_out);
	Tri output = input;
	for(int c = 0;c < input.size();++c)
	{
		output[c]=fpa.MAT_ADD_NT(input[c],conv2_out[c]);
	}
	return output;


}


//compute a fusion layer
//TEST:create new varible number to detect way to save resource
Tri Net::fusion_layer(const Tri& input1,const Tri& input2,const int C)
{
	Tri res_out1 = residual_block(input1);
	Tri add_result = fpa.TRI_ADD_NT(resout1,input2);
	Tri res_out2 = residual_block(add_result);
	Tri upsample_out = fpa.TRI_B_INTERPLO_NT(res_out2,(flaot)2.0);
	add_conv_layer(C,C/2,1,1,false);//add convlayer maybe need to be rewritten
	Tri output = compute_conv_layer(upsampled);
	return output;

}

//init a Lite-net
void initlitenet()
{
	add_conv_layer(3,32,3,2,true);
	block_configs =
	{
		{32,64,6,1},
		{64,128,6,2},
		{128,256,6,2},
		{256,512,6,1}
	};

	for(auto& [i_ch,o_ch,expand,stride]:block_configs)
	{
		add_efficient_block(i_ch,o_ch,expand,stride);
	}
}
void add_efficient_block(int i_ch,int o_ch,int expand,int stride)
{
	add_conv_layer(i_ch,i_ch * expand,1,1,false);
	add_conv_layer(i_ch * expand,i_ch * expand,3,stride,true);
	add_conv_layer(i_ch * expand,o_ch,1,1,false);
}




//compute a eln net
Tri Net::efficient_net(const Tri& input)
{
	Tri x = input;
	//three conv follow
	x = compute_cpnv_layer(x);
	x = compute_cpnv_layer(x);
	x = compute_cpnv_layer(x);
	x = fpa.TRI_ADD_NT(x,input);
	return x;

}






























