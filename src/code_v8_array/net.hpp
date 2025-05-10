#ifndef NET_H
#define NET_H

#include <iostream>
#include <array>
#include <algorithm>
#include "xfxpt_algebra.hpp"
#include <stdexcept>
#include <tuple>
#include <type_traits>

#define W 4
#define I 2
#define CF ap_fixed<W,I>
#define CFL ap_fixed<(2*W),(2*I)>

#define I_W 256
#define I_H 256
#define K_3 3
#define K_1 1
#define P_1 true
#define P_2 false
#define S_1 1
#define S_2 2

// 预定义网络参数（示例值，需根据实际模型确定）
constexpr size_t I_CH = 3;    // 输入通道
constexpr size_t I_H = 256;   // 输入高度
constexpr size_t I_W = 256;   // 输入宽度


constexpr size_t KERNEL_SIZE_3X3 = 3; // 卷积核尺寸3X3
constexpr size_t KERNEL_SIZE_1X1 = 1; // 卷积核尺寸1X1


constexpr size_t LAYERS_NUM_3X3 = 4; // 卷积层数量
constexpr size_t LAYERS_NUM_1X1 = 4; // 卷积层数量



// 类型别名模板化
template<typename T,size_t N> using Vec = std::array<T, N>;
template<typename T,size_t N> using Bias = std::array<T, N>;
template<typename T,size_t R, size_t C> using Mat = std::array<Vec<T,C>, R>;
template<typename T,size_t C, size_t H, size_t W> using Tri = std::array<Mat<T,H, W>, C>;
template<typename T,size_t OC, size_t IC, size_t K1,size_t K2> using ConvK = std::array<Tri<T,IC,K1,K2>,OC>;

// 网络配置类型
using ConvConfig = std::tuple<size_t, size_t, size_t, size_t, bool>;
using LiteConfig = std::tuple<size_t, size_t, size_t, size_t>;//(i_ch,o_ch,expand_ratio,stride)

namespace liu {
class Net {
public:
    // 使用预定义尺寸的array类型成员
    Tri<I_CH, I_H, I_W> input_buffer;
    Tri<I_CH, I_H, I_W> output_buffer;
    
    // 卷积相关成员 卷积核不使用向量表示了，使用单个array	后续按照网络结构补充
    std::array<ConvK<32, I_CH, KERNEL_SIZE_3X3> LAYERS_NUM_3X3> conv_kernels_3x3; // 32个输出通道
	std::array<ConvK<32, I_CH, KERNEL_SIZE_3X3> LAYERS_NUM_3X3> conv_kernels_3x3; // 32个输出通道
	
	
	// 卷积相关成员 卷积偏置也同样不使用向量表示，使用单个array	后续按照网络结构补充
    std::array<Vec<32>, LAYERS_NUM> conv_biases;
    std::array<ConvConfig, LAYERS_NUM> conv_configs;

    // 高效网络配置
    std::array<LiteConfig, 4> lite_configs {{
        {3,32,6,1},
        {32,48,6,2},
        {48,136,6,2},
        {136,384,6,1}
    }};


	//
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
	
	//void compute_conv_layer(const Tri (&input));
		
	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
	void compute_conv_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
					    Tri<T,o_ch,o_row,o_col>& FigB,
						const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel,
						const Bias<T,o_ch>& bias,
						);
	
	

	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
	void residual_block(const Tri<T,i_ch,i_row,i_col>& FigA,
							 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel1,
							 const Bias<T,o_ch>& bias1,
							 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel2,
							 const Bias<T,o_ch>& bias2,
							 Tri<T,o_ch,o_row,o_col>& FigB);
							
	
	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
	void fusion_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
						   const Tri<T,i_ch,i_row,i_col>& FigB,
						   //res1
						   const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel11,
						   const Bias<T,o_ch>& bias11,
						   const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel12,
						   const Bias<T,o_ch>& bias12,
						   ////////
						   //res2
						   const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel21,
						   const Bias<T,o_ch>& bias21,
						   const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel22,
						   const Bias<T,o_ch>& bias22,
						   ////////
						   Tri<T,o_ch,o_row,o_col>& FigC;
						   );


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
							
							 Tri<T,o_ch,o_row,o_col>& FigB);
	

	template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,size_t stride_1,size_t stride_2,size_t expand>
	void forward();


		//void initlitenet();

		//void add_efficient_block(int i_ch,int o_ch,int expand,int stride);

		Tri forward(Tri& input);

		void backward();

    // 其他成员函数需类似调整...
};

} // namespace liu
#endif




	protected:
		//init single layer's weight
		template<class T,int M,int N>
		Mat initweight_and_bias();
		//compute delta error
		void deltaError();
		//update weights
		void updateWeights();