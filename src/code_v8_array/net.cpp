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
	add_conv_layer(input.size(),input.size(),3,2,1);
	Tri conv1_out = compute_conv_layer(relu1_out);
	//2nd relu
	for (auto& channel : conv1_out)
	{
		fpa.RELU_MAT(channel);
	}
	//2nd conv 3x3
	add_conv_layer(input.size(),input.size(),3,2,1);
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
		{3,32,6,1},
		{32,48,6,2},
		{48,136,6,2},
		{136,384,6,1}
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
	x = compute_conv_layer(x);
	x = compute_conv_layer(x);
	x = compute_conv_layer(x);
	x = fpa.TRI_ADD_NT(x,input);
	return x;
}

//define forward
Tri Net::forward(Tri& input)
{
	initlitenet();
	Tri input_buffer = input;
	Tri x1,x2,x3,x4;
	Tri c1,c2,c3,c4;

	x1 = efficient_net(input_buffer);
	x2 = efficient_net(x1);
	x3 = efficient_net(x2);
	x4 = efficient_net(x3);

	add_conv_layer(32,64,3,2,1);
	c1 = compute_conv_layer(x1);

	add_conv_layer(48,128,3,2,1);
	c2 = compute_conv_layer(x2);


	add_conv_layer(136,256,3,2,1);
	c3 = compute_conv_layer(x3);

	add_conv_layer(384,512,3,2,1);
	c4 = compute_conv_layer(x4);

	Tri f1,f2,f3,f4;

	f4 = fusion_layer(c4,c4,512);
	f3 = fusion_layer(c3,f4,256);
	f2 = fusion_layer(c2,f3,128);
	f1 = fusion_layer(c1,f2,64);

	Tri o1,o2,o3,o4;
	add_conv_layer(32,32,3,2,1);
	o1 = compute_conv_layer(f1);

	o2 = fpa.TRI_B_INTERPLO_NT(o1);

	add_conv_layer(32,32,3,2,1);
	o3 = compute_conv_layer(o2);

	fpa.RELU_TRI(o3);

	add_conv_layer(32,1,1,1,0);
	o4 = compute_conv_layer(o3);

	fpa.RELU_TRI(o4);

	return o4;








}

//0505
//compute a full conv_layer
template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
void Net::compute_conv_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
						Tri<T,o_ch,o_row,o_col>& FigB,
						const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel,
						const Bias<T,o_ch>& bias,
						)
{
	fpa.CONV_FIG<T,i_ch,o_ch,i_row,o_row,i_col,o_col,kernel_size>(FigA,FigB,kernel,bias,padding,stride);						
}
//显式实例化Explicit instantiation
//input conv
template void Net::compute_conv_layer<CF,3,32,256,256,32,32,3,true,2>
(const Tri<CF,3,256,256>&,
	   Tri<CF,32,(256 + 2 * 1 - 3)/2 +1,(256 + 2 * 1 - 3)/2 +1>&,
 const ConvK<CF,32,3,3,3>&,
 const Bias<CF,32>&,
 

template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
void Net::residual_block(const Tri<T,i_ch,i_row,i_col>& FigA,
						 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel1,
						 const Bias<T,o_ch>& bias1,
						 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel2,
						 const Bias<T,o_ch>& bias2,
						 Tri<T,o_ch,o_row,o_col>& FigB)
						 {
							 Tri<T,i_ch,i_row,i_col> result1,result2,output;
							 fpa.RELU_Container<Tri<T,i_ch,i_row,i_col>>(FigA);
							 compute_conv_layer<T, i_ch,o_ch,i_row,i_col,o_row,o_col,kernel_size,padding,stride>(FigA,result1,kernel1,bias1);
							 fpa.RELU_Container<Tri<T,i_ch,i_row,i_col>>(result1);
							 compute_conv_layer<T, i_ch,o_ch,i_row,i_col,o_row,o_col,kernel_size,padding,stride>(result1,result2,kernel2,bias2);
							 fpa.TRI_ADD<T,i_ch,i_row,i_col>(result2,FigA,output);
						 }


	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,int stride_1,int stride_2>
	void Net::fusion_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
						   const Tri<T,i_ch,i_row,i_col>& FigB,
						   //res1
						   const Convk<T,i_ch,i_ch,kernel_size_3x3,kernel_size_3x3>& kernel11,
						   const Bias<T,i_ch>& bias11,
						   const Convk<T,i_ch,i_ch,kernel_size_3x3,kernel_size_3x3>& kernel12,
						   const Bias<T,i_ch>& bias12,
						   //res2
						   const Convk<T,i_ch,i_ch,kernel_size_3x3,kernel_size_3x3>& kernel21,
						   const Bias<T,i_ch>& bias21,
						   const Convk<T,i_ch,i_ch,kernel_size_3x3,kernel_size_3x3>& kernel22,
						   const Bias<T,i_ch>& bias22,
						   ////////
						   //conv 1x1
						   const Convk<T,o_ch,i_ch,kernel_size_1x1,kernel_size_1x1>& kernel3,
						   const Bias<T,o_ch>& bias3,
						   /////////
						   Tri<T,o_ch,o_row,o_col>& FigC;
						   )
						   {
							   Tri<T,i_ch,i_row,i_col> res_result_1,add_result_1,add_result_2;
							   Tri<T,o_ch,o_row,o_col> bl_result_1
							   //第一次残差res1 input 与 output 的尺寸相同
							   residual_block<T, i_ch,i_ch,i_row,i_col,i_row,i_col,kernel_size_3x3,padding_1,stride_2>(FigA,kernel11,bias11,kernel12,bias12,res_result_1);
							   fpa.TRI_ADD<T,i_ch,i_row,i_col>(res_result_1,FigB,add_result_1);
							   //第二次res2 input 与 output尺寸仍然相同
							   residual_block<T, i_ch,i_ch,i_row,i_col,i_row,i_col,kernel_size_3x3,padding_1,stride_2>(add_result_1,kernel21,bias21,kernel22,bias22,add_result_2);
							   fpa.BILINEAR_INTERPOLATION_TRI<T,i_ch,i_row,i_col,o_row,o_col>(add_result_2,bl_result_1);
							   //这个卷积 使用1X1卷积核 输入输出尺寸固定
							   compute_conv_layer<T, i_ch,o_ch,o_row,o_col,o_row,o_col,kernel_size_1x1,padding_2,stride_2>(bl_result_1,FigC,kernel3,bias3);
							   
	
							   
						   }



	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,size_t stride_1,size_t stride_2,size_t expand>
	void efficient_block(const Tri<T,i_ch,i_row,i_col>& FigA,
							//conv1
							 const Convk<T,i_ch*expand,i_ch,kernel_size_1x1,kernel_size_1x1>& kernel1,
							 const Bias<T,i_ch*expand>& bias1,
							//conv2
							 const Convk<T,i_ch*expand,i_ch*expand,kernel_size_3x3,kernel_size_3x3>& kernel2,
							 const Bias<T,i_ch*expand>& bias2,
							//conv3
							 const Convk<T,o_ch,i_ch*expand,kernel_size_1x1,kernel_size_1x1>& kernel3,
							 const Bias<T,o_ch>& bias3,
							
							 Tri<T,o_ch,o_row,o_col>& FigB)
							 {
								 Tri <T,i_ch*expand,i_row,i_col> result1,result2;
								 Tri <T,o_ch,i_row,i_col> result3;
								 compute_conv_layer<T,i_ch,i_ch*expand,i_row,i_col,o_row,o_col,kernel_size_1x1,padding_2,stride_2>(FigA,result1,kernel1,bias1);
								 compute_conv_layer<T,i_ch*expand,i_ch*expand,i_row,i_col,o_row,o_col,kernel_size_3x3,padding_1,stride_1>(result1,result2,kernel2,bias2);
								 compute_conv_layer<T,i_ch*expand,o_c,i_row,i_col,o_row,o_col,kernel_size_1x1,padding_2,stride_2>(result2,result3,kernel3,bias3);
							 }
							 


#define I_W 256
#define I_H 256
#define K_3 3
#define K_1 1
#define P_1 true
#define P_2 false
#define S_1 1
#define S_2 2

		//每个3x3卷积模块/efficient-net让特征图尺寸减半
		//efficient-net-1 kernels
		Convk<T,3*expand,3,kernel_size_1x1,kernel_size_1x1> kernel_e1_1;
		Convk<T,3*expand,3*expand,kernel_size_3x3,kernel_size_3x3> kernel_e1_2;
		Convk<T,32,3*expand,kernel_size_1x1,kernel_size_1x1> kernel_e1_3;
		Bias<T,32> bias_e1_1;
		Bias<T,32> bias_e1_2;
		Bias<T,32> bias_e1_3;
		//efficient-net-2 kernels
		Convk<T,32*expand,32,kernel_size_1x1,kernel_size_1x1> kernel_e2_1;
		Convk<T,32*expand,32*expand,kernel_size_3x3,kernel_size_3x3> kernel_e2_2;
		Convk<T,48,32*expand,kernel_size_1x1,kernel_size_1x1> kernel_e2_3;
		Bias<T,48> bias_e2_1;
		Bias<T,48> bias_e2_2;
		Bias<T,48> bias_e2_3;
		//efficient-net-3 kernels
		Convk<T,48*expand,48,kernel_size_1x1,kernel_size_1x1> kernel_e3_1;
		Convk<T,48*expand,48*expand,kernel_size_3x3,kernel_size_3x3> kernel_e3_2;
		Convk<T,136,48*expand,kernel_size_1x1,kernel_size_1x1> kernel_e3_3;
		Bias<T,136> bias_e3_1;
		Bias<T,136> bias_e3_2;
		Bias<T,136> bias_e3_3;
		//efficient-net-4 kernels
		Convk<T,136*expand,136,kernel_size_1x1,kernel_size_1x1> kernel_e4_1;
		Convk<T,136*expand,136*expand,kernel_size_3x3,kernel_size_3x3> kernel_e4_2;
		Convk<T,384,136*expand,kernel_size_1x1,kernel_size_1x1> kernel_e4_3;
		Bias<T,384> bias_e4_1;
		Bias<T,384> bias_e4_2;
		Bias<T,384> bias_e4_3;
		
		//encoder_conv不降低特征图尺寸
		//encoder kernel
		Convk<T,64,32,kernel_size_3x3,kernel_size_3x3> kernel_en_conv_1;
		Bias<T,64> bias_en_conv_1;
		Convk<T,128,48,kernel_size_3x3,kernel_size_3x3> kernel_en_conv_2;
		Bias<T,128> bias_en_conv_2;
		Convk<T,256,136,kernel_size_3x3,kernel_size_3x3> kernel_en_conv_3;
		Bias<T,256> bias_en_conv_3;
		Convk<T,512,384,kernel_size_3x3,kernel_size_3x3> kernel_en_conv_4;
		Bias<T,512> bias_en_conv_4;
		
		//fusion_conv kernel
		Convk<T,64,64,kernel_size_3x3,kernel_size_3x3> kernel_fusion1_res_1_conv1;
		Bias<T,64> bias_fusion1_res_1_conv1;
		Convk<T,64,64,kernel_size_3x3,kernel_size_3x3> kernel_fusion1_res_1_conv2;
		Bias<T,64> bias_fusion1_res_1_conv2;
		Convk<T,64,64,kernel_size_3x3,kernel_size_3x3> kernel_fusion1_res_2_conv1;
		Bias<T,64> bias_fusion1_res_2_conv1;
		Convk<T,64,64,kernel_size_3x3,kernel_size_3x3> kernel_fusion1_res_2_conv2;
		Bias<T,64> bias_fusion1_res_2_conv2;
		Convk<T,64,64,kernel_size_3x3,kernel_size_3x3> kernel_fusion1_conv3;
		Bias<T,64> bias_fusion1_conv3;
		
		Convk<T,128,128,kernel_size_3x3,kernel_size_3x3> kernel_fusion2_res_1_conv1;
		Bias<T,128> bias_fusion2_res_1_conv1;
		Convk<T,128,128,kernel_size_3x3,kernel_size_3x3> kernel_fusion2_res_1_conv2;
		Bias<T,128> bias_fusion2_res_1_conv2;
		Convk<T,128,128,kernel_size_3x3,kernel_size_3x3> kernel_fusion2_res_2_conv1;
		Bias<T,128> bias_fusion2_res_2_conv1;
		Convk<T,128,128,kernel_size_3x3,kernel_size_3x3> kernel_fusion2_res_2_conv2;
		Bias<T,128> bias_fusion2_res_2_conv2;
		Convk<T,64,128,kernel_size_3x3,kernel_size_3x3> kernel_fusion2_conv3;
		Bias<T,64> bias_fusion2_conv3;
		
		Convk<T,256,256,kernel_size_3x3,kernel_size_3x3> kernel_fusion3_res_1_conv1;
		Bias<T,256> bias_fusion3_res_1_conv1;
		Convk<T,256,256,kernel_size_3x3,kernel_size_3x3> kernel_fusion3_res_1_conv2;
		Bias<T,256> bias_fusion3_res_1_conv2;
		Convk<T,256,256,kernel_size_3x3,kernel_size_3x3> kernel_fusion3_res_2_conv1;
		Bias<T,256> bias_fusion3_res_2_conv1;
		Convk<T,256,256,kernel_size_3x3,kernel_size_3x3> kernel_fusion3_res_2_conv2;
		Bias<T,256> bias_fusion3_res_2_conv2;
		Convk<T,128,256,kernel_size_3x3,kernel_size_3x3> kernel_fusion3_conv3;
		Bias<T,128> bias_fusion3_conv3;
		
		Convk<T,512,512,kernel_size_3x3,kernel_size_3x3> kernel_fusion4_res_1_conv1;
		Bias<T,512> bias_fusion4_res_1_conv1;
		Convk<T,512,512,kernel_size_3x3,kernel_size_3x3> kernel_fusion4_res_1_conv2;
		Bias<T,512> bias_fusion4_res_1_conv2;
		Convk<T,512,512,kernel_size_3x3,kernel_size_3x3> kernel_fusion4_res_2_conv1;
		Bias<T,512> bias_fusion4_res_2_conv1;
		Convk<T,512,512,kernel_size_3x3,kernel_size_3x3> kernel_fusion4_res_2_conv2;
		Bias<T,512> bias_fusion4_res_2_conv2;
		Convk<T,512,512,kernel_size_3x3,kernel_size_3x3> kernel_fusion4_conv3;
		Bias<T,256> bias_fusion4_conv3;
		
		//OutputConv kernel
		Convk<T,32,64,kernel_size_3x3,kernel_size_3x3> outputconv_1;
		Bias<T,32> bias_outputconv_1;
		Convk<T,32,32,kernel_size_3x3,kernel_size_3x3> outputconv_2;
		Bias<T,32> bias_outputconv_2;
		Convk<T,1,32,kernel_size_3x3,kernel_size_3x3> outputconv_3;
		Bias<T,1> bias_outputconv_3;
		/////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////
		//input buffer
		Tri<T,3,256,256> inputbuffer;
		//efficient-net-buffer
		Tri<T,32,128,128> eff_buf_1;
		Tri<T,48,64,64> eff_buf_2;
		Tri<T,136,32,32> eff_buf_3;
		Tri<T,384,16,16> eff_buf_4;
		//encoder_conv_buffer
		Tri<T,64,128,128> encoder_conv_buf_1;
		Tri<T,128,64,64> encoder_conv_buf_2;
		Tri<T,256,32,32> encoder_conv_buf_3;
		Tri<T,512,16,16> encoder_conv_buf_4;
		//fusion1_buffer
		Tri<T,64,256,256> fusion1_buf;
		//fusion2_buffer
		Tri<T,64,128,128> fusion2_buf;
		//fusion3_buffer
		Tri<T,128,64,64> fusion3_buf;
		//fusion4_buffer
		Tri<T,256,32,32> fusion4_buf;
		//OutputConv
		Tri<T,32,256,256> outputconv_buf_1;
		Tri<T,32,256,256> outputconv_buf_2;
		Tri<T,1,256,256> outputconv_buf_3;

		

	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,size_t stride_1,size_t stride_2,size_t expand>
	void Net::forward()
	{
		//efficient-net
		efficient_block<T,3,32,I_H,I_W,I_H/2,I_W/2,K_3,K_1,P_1,P_2,S_2,S_1,expand>(inputbuffer,kernel_e1_1,bias_e1_1,kernel_e1_2,bias_e1_2,kernel_e1_3,bias_e1_3,eff_buf_1);
		efficient_block<T,32,48,I_H/2,I_W/2,I_H/4,I_W/4,K_3,K_1,P_1,P_2,S_2,S_1,expand>(eff_buf_1,kernel_e2_1,bias_e2_1,kernel_e2_2,bias_e2_2,kernel_e2_3,bias_e2_3,eff_buf_2);
		efficient_block<T,48,136,I_H/4,I_W/4,I_H/8,I_W/8,K_3,K_1,P_1,P_2,S_2,S_1,expand>(eff_buf_2,kernel_e3_1,bias_e3_1,kernel_e3_2,bias_e3_2,kernel_e3_3,bias_e3_3,eff_buf_3);
		efficient_block<T,136,384,I_H/8,I_W/8,I_H/16,I_W/16,K_3,K_1,P_1,P_2,S_2,S_1,expand>(eff_buf_3,kernel_e4_1,bias_e4_1,kernel_e4_2,bias_e4_2,kernel_e4_3,bias_e4_3,eff_buf_4);
		
		//encoder
		compute_conv_layer<T,32,64,128,128,128,128,K_3,P_1,S_1>(eff_buf_1,encoder_conv_buf_1,kernel_en_conv_1,bias_en_conv_1);
		compute_conv_layer<T,48,128,64,64,64,64,K_3,P_1,S_1>(eff_buf_2,encoder_conv_buf_2,kernel_en_conv_2,bias_en_conv_2);
		compute_conv_layer<T,136,256,32,32,32,32,K_3,P_1,S_1>(eff_buf_3,encoder_conv_buf_3,kernel_en_conv_3,bias_en_conv_3);
		compute_conv_layer<T,384,512,16,16,16,16,K_3,P_1,S_1>(eff_buf_4,encoder_conv_buf_4,kernel_en_conv_4,bias_en_conv_4);

		//fusion4
		fusion_layer<T,512,256,16,16,32,32,K_3,K_1,P_1,P_2,S_1,S_2>(encoder_conv_buf_4,encoder_conv_buf_4,kernel_fusion4_res_1_conv1,bias_fusion4_res_1_conv1,kernel_fusion4_res_1_conv2,
		kernel_fusion4_res_2_conv1,bias_fusion4_res_2_conv1,kernel_fusion4_res_2_conv2,bias_fusion4_res_2_conv2,kernel_fusion4_conv3,bias_fusion4_conv3,fusion4_buf);
		
		//fusion3
		fusion_layer<T,256,128,32,32,64,64,K_3,K_1,P_1,P_2,S_1,S_2>(encoder_conv_buf_3,fusion4_buf,kernel_fusion3_res_1_conv1,bias_fusion3_res_1_conv1,kernel_fusion3_res_1_conv2,
		kernel_fusion3_res_2_conv1,bias_fusion3_res_2_conv1,kernel_fusion3_res_2_conv2,bias_fusion3_res_2_conv2,kernel_fusion3_conv3,bias_fusion3_conv3,fusion3_buf);
		
		//fusion2
		fusion_layer<T,128,64,64,64,128,128,K_3,K_1,P_1,P_2,S_1,S_2>(encoder_conv_buf_2,fusion3_buf,kernel_fusion2_res_1_conv1,bias_fusion2_res_1_conv1,kernel_fusion2_res_1_conv2,
		kernel_fusion2_res_2_conv1,bias_fusion2_res_2_conv1,kernel_fusion2_res_2_conv2,bias_fusion2_res_2_conv2,kernel_fusion2_conv3,bias_fusion2_conv3,fusion2_buf);
		
		//fusion1
		fusion_layer<T,64,64,128,128,256,256,K_3,K_1,P_1,P_2,S_1,S_2>(encoder_conv_buf_1,fusion2_buf,kernel_fusion1_res_1_conv1,bias_fusion1_res_1_conv1,kernel_fusion1_res_1_conv2,
		kernel_fusion1_res_2_conv1,bias_fusion1_res_2_conv1,kernel_fusion1_res_2_conv2,bias_fusion1_res_2_conv2,kernel_fusion1_conv3,bias_fusion1_conv3,fusion1_buf);
		
		//outputconv_1
		compute_conv_layer<T,64,32,256,256,256,256,K_3,P_1,S_1>(fusion1_buf,outputconv_buf_1,outputconv_1,bias_outputconv_1);
		compute_conv_layer<T,32,32,256,256,256,256,K_3,P_1,S_1>(outputconv_buf_1,outputconv_buf_2,outputconv_2,bias_outputconv_2);
		fpa.RELU_Container<T>(outputconv_buf_2);
		compute_conv_layer<T,32,1,256,256,256,256,K_3,P_1,S_1>(outputconv_buf_2,outputconv_buf_3,outputconv_3,bias_outputconv_3);
		fpa.RELU_Container<T>(outputconv_buf_3);
	}

























