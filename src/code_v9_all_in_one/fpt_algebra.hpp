/*
 *	This is light floating point algebra library
 *	Author: Yun Wu
 *	Created by: 2019-06-13
 *	Copyright @ Yun Wu
 *
 */

#ifndef SRC_FPT_ALGEBRA_HPP_
#define SRC_FPT_ALGEBRA_HPP_

#include "data.hpp"
#include "common.hpp"

#include <iostream>
#include <array>
#include <algorithm>

#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <ap_fixed.h>

#define W 4
#define I 2
#define CF ap_fixed<W,I>
#define CFL ap_fixed<(2*W),(2*I)>

#define K_3 3
#define K_1 1
#define P_1 true
#define P_2 false
#define S_1 1
#define S_2 2

// 预定义网络参数（示例值，需根据实际模型确定）
constexpr size_t I_CH = 3;   // 输入通道




constexpr size_t I_H = 256;    // 输入高度
constexpr size_t I_W = 256;  // 输入宽度


constexpr size_t KERNEL_SIZE_3X3 = 3; // 卷积核尺寸3X3
constexpr size_t KERNEL_SIZE_1X1 = 1; // 卷积核尺寸1X1


constexpr size_t LAYERS_NUM_3X3 = 4; // 卷积层数量
constexpr size_t LAYERS_NUM_1X1 = 4; // 卷积层数量



// 类型别名模板化
template<typename T,size_t TN> using Vec = std::array<T, TN>;
template<typename T,size_t N> using Bias = std::array<T, N>;
template<typename T,size_t R, size_t C> using Mat = std::array<std::array<T,C>, R>;
template<typename T,size_t TH, size_t TR,size_t TC> using Tri = std::array<std::array<std::array<T,TC>, TR>,TH>;
template<typename T,size_t OC, size_t IC, size_t K1,size_t K2> using Convk = std::array<Tri<T,IC,K1,K2>,OC>;

// 网络配置类型
using ConvConfig = std::tuple<size_t, size_t, size_t, size_t, bool>;
using LiteConfig = std::tuple<size_t, size_t, size_t, size_t>;//(i_ch,o_ch,expand_ratio,stride)















class Float_Point_Algebra {

public:

	
	// 0427 ADD Matrix conv



	//0429 add some operate without template

	//add random vec without template
	//using & to change value in vector

	//compile error


	
//////////////////////////////////////////////////////////////////////////
//use array and template	
//0504
//////////////////////////////////////////////////////////////////////////


//template Tri conv 0505
template <typename T,size_t i_ch,size_t o_ch,size_t i_row,size_t o_row,size_t i_col,size_t o_col,size_t kernel_size,bool padding,int stride>
void CONV_FIG(const Tri<T,i_ch,i_row,i_col>& FigA,
			  Tri<T,o_ch,o_row,o_col>&FigB,
			  const Convk<T,o_ch,i_ch,kernel_size,kernel_size>&kernel,
			  const Bias<T,o_ch>&bias
)
	{
		constexpr int Padding = padding ? 1 : 0;
//		static_assert(o_row == (i_row + 2 * Padding - kernel_size)/stride +1,"Conv output row dismatch!");
//		static_assert(o_col == (i_col + 2 * Padding - kernel_size)/stride +1,"Conv output col dismatch!");
		//conv operation
		
		for (int c = 0; c < o_ch; c++) {
			for (int out_i = 0; out_i < o_row; out_i += stride) {
				for (int out_j = 0; out_j < o_col; out_j += stride) {
					T sum = 0;
					for (int in_ch = 0; in_ch < i_ch; in_ch++) {
						for (int k_i = 0; k_i < kernel_size; k_i +=stride) {
							for (int k_j = 0; k_j < kernel_size; k_j +=stride) {
								int in_i = out_i + k_i;
								int in_j = out_j + k_j;
								if (in_i < 0 || in_i > i_row || in_j < 0
										|| in_j > i_col) {
									sum = sum;
								} else {
									sum = sum+ FigA[in_ch][in_i][in_j]* kernel[c][in_ch][k_i][k_j];
								}
							}
						}
					}
					FigB[c][out_i][out_j] = sum + bias[c];
				}
			}
		}
		
		
	}
	
//template RELU iteration
template <typename T,size_t ch,size_t row,size_t col>
void RELU_Container(const Tri<T,ch,row,col> &FigA,Tri<T,ch,row,col>&FigB)
{
    for(int i = 0;i<ch;++i){
    	for(int j = 0;j<row;++j){
    		for(int k = 0;k<col;++j){
    			if(FigA[i][j][k] > 0)
    			{
    				FigB[i][j][k] = FigA[i][j][k];
    			}
    			else
    			{
    				FigB[i][j][k] = 0;
    			}
    		}
    	}
    }

}

//template Tri bilinear_interpolation 

// 二维矩阵双线性插值模板
template<class T, int in_row, int in_col, int o_row, int o_col>
void BILINEAR_INTERPOLATION_2D(const Mat<T,in_row,in_col> &input, Mat<T,o_row,o_col> &output) {
    constexpr float x_scale = (in_row > 1) ? (in_row - 1.0f) / (o_row - 1.0f) : 0.0f;
    constexpr float y_scale = (in_col > 1) ? (in_col - 1.0f) / (o_col - 1.0f) : 0.0f;

    for (int y = 0; y < o_col; ++y) {
        const float src_y = y * y_scale;
        const int y0 = std::max(0, std::min(static_cast<int>(src_y), in_col - 1));
        const int y1 = std::max(0, std::min(y0 + 1, in_col - 1));
        const T dy = src_y - y0;

        for (int x = 0; x < o_row; ++x) {
            const float src_x = x * x_scale;
            const int x0 = std::max(0, std::min(static_cast<int>(src_x), in_row - 1));
            const int x1 = std::max(0, std::min(x0 + 1, in_row - 1));
            const T dx = src_x - x0;

            // 双线性插值计算
            const T& val00 = input[x0][y0];
            const T& val01 = input[x0][y1];
            const T& val10 = input[x1][y0];
            const T& val11 = input[x1][y1];

            const T val0 = val00 + dx * (val10 - val00);
            const T val1 = val01 + dx * (val11 - val01);
            output[x][y] = val0 + dy * (val1 - val0);
        }
    }
}

// 三维Tensor双线性插值模板
template<class T, int C, int in_row, int in_col, int o_row, int o_col>
void BILINEAR_INTERPOLATION_TRI(const Tri<T,C,in_row,in_col> &input, Tri<T,C,o_row,o_col> &output) {
    // 对每个通道进行独立插值
    for (int ch = 0; ch < C; ++ch) {
        BILINEAR_INTERPOLATION_2D<T, in_row, in_col, o_row, o_col>(input[ch], output[ch]);
    }
}

// 带缩放因子的通用版本
template<class T, int C, int in_row, int in_col, int o_row, int o_col, int scale>
void BILINEAR_INTERPOLATION_TRI_SCALE(const Tri<T,C,in_row,in_col> &input, Tri<T,C,o_row,o_col> &output) {
//    static_assert(o_row == in_row * scale, "Output height mismatch");
//    static_assert(o_col == in_col * scale, "Output width mismatch");
    BILINEAR_INTERPOLATION_TRI<T, C, in_row, in_col, o_row, o_col>(input, output);
}


//矩阵元素相加 用于三维特征值
template<class T, int _C, int _H, int _W>
void TRI_ADD(const Tri<T, _C, _H, _W>& A,
            const Tri<T, _C, _H, _W>& B,
            Tri<T, _C, _H, _W>& C) {
    // 静态断言确保维度匹配
//    static_assert(A.size() == C && B.size() == C,
//                "Channel dimension mismatch!");
    
    // 三维张量逐元素相加
    for (int c = 0; c < _C; ++c) {
        for (int h = 0; h < _H; ++h) {
            for (int w = 0; w < _W; ++w) {
                // 使用带范围检查的访问 (如果底层容器支持)
                C[c][h][w] = A[c][h][w] + B[c][h][w]; 
            }
        }
    }
}



private:

protected:

};



class Net {
public:



	Float_Point_Algebra fpa;


	//
	//每个3x3卷积模块/efficient-net让特征图尺寸减半
		//efficient-net-1 kernels
		Convk<CF,3*6,3,1,1> kernel_e1_1;
		Convk<CF,3*6,3*6,3,3> kernel_e1_2;
		Convk<CF,32,3*6,1,1> kernel_e1_3;
		Bias<CF,18> bias_e1_1;
		Bias<CF,18> bias_e1_2;
		Bias<CF,32> bias_e1_3;
		//efficienCF-net-2 kernels
		Convk<CF,32*6,32,1,1> kernel_e2_1;
		Convk<CF,32*6,32*6,3,3> kernel_e2_2;
		Convk<CF,48,32*6,1,1> kernel_e2_3;
		Bias<CF,32*6> bias_e2_1;
		Bias<CF,32*6> bias_e2_2;
		Bias<CF,48> bias_e2_3;
		//efficient-net-3 kernels
		Convk<CF,48*6,48,1,1> kernel_e3_1;
		Convk<CF,48*6,48*6,3,3> kernel_e3_2;
		Convk<CF,136,48*6,1,1> kernel_e3_3;
		Bias<CF,48*6> bias_e3_1;
		Bias<CF,48*6> bias_e3_2;
		Bias<CF,136> bias_e3_3;
		//efficient-net-4 kernels
		Convk<CF,136*6,136,1,1> kernel_e4_1;
		Convk<CF,136*6,136*6,3,3> kernel_e4_2;
		Convk<CF,384,136*6,1,1> kernel_e4_3;
		Bias<CF,136*6> bias_e4_1;
		Bias<CF,136*6> bias_e4_2;
		Bias<CF,384> bias_e4_3;

		//encoder_conv不降低特征图尺寸
		//encoder kernel
		Convk<CF,64,32,3,3> kernel_en_conv_1;
		Bias<CF,64> bias_en_conv_1;
		Convk<CF,128,48,3,3> kernel_en_conv_2;
		Bias<CF,128> bias_en_conv_2;
		Convk<CF,256,136,3,3> kernel_en_conv_3;
		Bias<CF,256> bias_en_conv_3;
		Convk<CF,512,384,3,3> kernel_en_conv_4;
		Bias<CF,512> bias_en_conv_4;

		//fusion_conv kernel
		Convk<CF,64,64,3,3> kernel_fusion1_res_1_conv1;
		Bias<CF,64> bias_fusion1_res_1_conv1;
		Convk<CF,64,64,3,3> kernel_fusion1_res_1_conv2;
		Bias<CF,64> bias_fusion1_res_1_conv2;
		Convk<CF,64,64,3,3> kernel_fusion1_res_2_conv1;
		Bias<CF,64> bias_fusion1_res_2_conv1;
		Convk<CF,64,64,3,3> kernel_fusion1_res_2_conv2;
		Bias<CF,64> bias_fusion1_res_2_conv2;
		Convk<CF,64,64,1,1> kernel_fusion1_conv3;
		Bias<CF,64> bias_fusion1_conv3;

		Convk<CF,128,128,3,3> kernel_fusion2_res_1_conv1;
		Bias<CF,128> bias_fusion2_res_1_conv1;
		Convk<CF,128,128,3,3> kernel_fusion2_res_1_conv2;
		Bias<CF,128> bias_fusion2_res_1_conv2;
		Convk<CF,128,128,3,3> kernel_fusion2_res_2_conv1;
		Bias<CF,128> bias_fusion2_res_2_conv1;
		Convk<CF,128,128,3,3> kernel_fusion2_res_2_conv2;
		Bias<CF,128> bias_fusion2_res_2_conv2;
		Convk<CF,64,128,1,1> kernel_fusion2_conv3;
		Bias<CF,64> bias_fusion2_conv3;

		Convk<CF,256,256,3,3> kernel_fusion3_res_1_conv1;
		Bias<CF,256> bias_fusion3_res_1_conv1;
		Convk<CF,256,256,3,3> kernel_fusion3_res_1_conv2;
		Bias<CF,256> bias_fusion3_res_1_conv2;
		Convk<CF,256,256,3,3> kernel_fusion3_res_2_conv1;
		Bias<CF,256> bias_fusion3_res_2_conv1;
		Convk<CF,256,256,3,3> kernel_fusion3_res_2_conv2;
		Bias<CF,256> bias_fusion3_res_2_conv2;
		Convk<CF,128,256,1,1> kernel_fusion3_conv3;
		Bias<CF,128> bias_fusion3_conv3;

		Convk<CF,512,512,3,3> kernel_fusion4_res_1_conv1;
		Bias<CF,512> bias_fusion4_res_1_conv1;
		Convk<CF,512,512,3,3> kernel_fusion4_res_1_conv2;
		Bias<CF,512> bias_fusion4_res_1_conv2;
		Convk<CF,512,512,3,3> kernel_fusion4_res_2_conv1;
		Bias<CF,512> bias_fusion4_res_2_conv1;
		Convk<CF,512,512,3,3> kernel_fusion4_res_2_conv2;
		Bias<CF,512> bias_fusion4_res_2_conv2;
		Convk<CF,256,512,1,1> kernel_fusion4_conv3;
		Bias<CF,256> bias_fusion4_conv3;

		//OutputConv kernel
		Convk<CF,32,64,3,3> outputconv_1;
		Bias<CF,32> bias_outputconv_1;
		Convk<CF,32,32,3,3> outputconv_2;
		Bias<CF,32> bias_outputconv_2;
		Convk<CF,1,32,3,3> outputconv_3;
		Bias<CF,1> bias_outputconv_3;
		/////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////
		//input buffer
		Tri<CF,3,256,256> inputbuffer;
		//efficient-net-buffer
		Tri<CF,32,128,128> eff_buf_1;
		Tri<CF,48,64,64> eff_buf_2;
		Tri<CF,136,32,32> eff_buf_3;
		Tri<CF,384,16,16> eff_buf_4;
		//encoder_conv_buffer
		Tri<CF,64,128,128> encoder_conv_buf_1;
		Tri<CF,128,64,64> encoder_conv_buf_2;
		Tri<CF,256,32,32> encoder_conv_buf_3;
		Tri<CF,512,16,16> encoder_conv_buf_4;
		//fusion1_buffer
		Tri<CF,64,256,256> fusion1_buf;
		//fusion2_buffer
		Tri<CF,64,128,128> fusion2_buf;
		//fusion3_buffer
		Tri<CF,128,64,64> fusion3_buf;
		//fusion4_buffer
		Tri<CF,256,32,32> fusion4_buf;
		//OutputConv
		Tri<CF,32,256,256> outputconv_buf_1;
		Tri<CF,32,256,256> outputconv_buf_2;
		Tri<CF,32,256,256> outputconv_relu_buf_2;
		Tri<CF,1,256,256> outputconv_buf_3;
		Tri<CF,1,256,256> outputconv_relu_buf_3;

	//void compute_conv_layer(const Tri (&input));

		template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
		void compute_conv_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
								Tri<T,o_ch,o_row,o_col>& FigB,
								const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel,
								const Bias<T,o_ch>& bias
								)
		{
			fpa.CONV_FIG<T,i_ch,o_ch,i_row,o_row,i_col,o_col,kernel_size,padding,stride>(FigA,FigB,kernel,bias);
		}



		template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
		void residual_block(const Tri<T,i_ch,i_row,i_col>& FigA,
								 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel1,
								 const Bias<T,o_ch>& bias1,
								 const Convk<T,o_ch,i_ch,kernel_size,kernel_size>& kernel2,
								 const Bias<T,o_ch>& bias2,
								 Tri<T,o_ch,o_row,o_col>& FigB)
								 {
									 Tri<T,i_ch,i_row,i_col> result1,result2,relu_result1,relu_result2,output;
									 fpa.RELU_Container<T>(FigA,relu_result1);
									 compute_conv_layer<T, i_ch,o_ch,i_row,i_col,o_row,o_col,kernel_size,padding,stride>(FigA,result1,kernel1,bias1);
									 fpa.RELU_Container<T>(result1,relu_result2);
									 compute_conv_layer<T, i_ch,o_ch,i_row,i_col,o_row,o_col,kernel_size,padding,stride>(result1,result2,kernel2,bias2);
									 fpa.TRI_ADD<T,i_ch,i_row,i_col>(result2,FigA,output);
								 }


		template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,int stride_1,int stride_2>
		void fusion_layer(const Tri<T,i_ch,i_row,i_col>& FigA,
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
								   Tri<T,o_ch,o_row,o_col>& FigC
								   )
								   {
									   Tri<T,i_ch,i_row,i_col> res_result_1,add_result_1,add_result_2;
									   Tri<T,i_ch,o_row,o_col> bl_result_1;
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
										 Tri <T,i_ch*expand,i_row,i_col> result1;
										 Tri <T,i_ch*expand,o_row,o_col> result2;
										 Tri <T,o_ch,o_row,o_col> result3;
										 compute_conv_layer<T,i_ch,i_ch*expand,i_row,i_col,i_row,i_col,kernel_size_1x1,padding_2,stride_1>(FigA,result1,kernel1,bias1);
										 compute_conv_layer<T,i_ch*expand,i_ch*expand,i_row,i_col,o_row,o_col,kernel_size_3x3,padding_1,stride_2>(result1,result2,kernel2,bias2);
										 compute_conv_layer<T,i_ch*expand,o_ch,o_row,o_col,o_row,o_col,kernel_size_1x1,padding_2,stride_1>(result2,result3,kernel3,bias3);
									 }


		template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,size_t stride_1,size_t stride_2,size_t expand>
			Tri<T,1,256,256> forward()
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
				fusion_layer<T,512,256,16,16,32,32,K_3,K_1,P_1,P_2,S_2,S_1>(encoder_conv_buf_4,encoder_conv_buf_4,kernel_fusion4_res_1_conv1,bias_fusion4_res_1_conv1,kernel_fusion4_res_1_conv2,bias_fusion4_res_1_conv2,
				kernel_fusion4_res_2_conv1,bias_fusion4_res_2_conv1,kernel_fusion4_res_2_conv2,bias_fusion4_res_2_conv2,kernel_fusion4_conv3,bias_fusion4_conv3,fusion4_buf);

				//fusion3
				fusion_layer<T,256,128,32,32,64,64,K_3,K_1,P_1,P_2,S_2,S_1>(encoder_conv_buf_3,fusion4_buf,kernel_fusion3_res_1_conv1,bias_fusion3_res_1_conv1,kernel_fusion3_res_1_conv2,bias_fusion3_res_1_conv2,
				kernel_fusion3_res_2_conv1,bias_fusion3_res_2_conv1,kernel_fusion3_res_2_conv2,bias_fusion3_res_2_conv2,kernel_fusion3_conv3,bias_fusion3_conv3,fusion3_buf);

				//fusion2
				fusion_layer<T,128,64,64,64,128,128,K_3,K_1,P_1,P_2,S_2,S_1>(encoder_conv_buf_2,fusion3_buf,kernel_fusion2_res_1_conv1,bias_fusion2_res_1_conv1,kernel_fusion2_res_1_conv2,bias_fusion2_res_1_conv2,
				kernel_fusion2_res_2_conv1,bias_fusion2_res_2_conv1,kernel_fusion2_res_2_conv2,bias_fusion2_res_2_conv2,kernel_fusion2_conv3,bias_fusion2_conv3,fusion2_buf);

				//fusion1
				fusion_layer<T,64,64,128,128,256,256,K_3,K_1,P_1,P_2,S_2,S_1>(encoder_conv_buf_1,fusion2_buf,kernel_fusion1_res_1_conv1,bias_fusion1_res_1_conv1,kernel_fusion1_res_1_conv2,bias_fusion1_res_1_conv2,
				kernel_fusion1_res_2_conv1,bias_fusion1_res_2_conv1,kernel_fusion1_res_2_conv2,bias_fusion1_res_2_conv2,kernel_fusion1_conv3,bias_fusion1_conv3,fusion1_buf);

				//outputconv_1
				compute_conv_layer<T,64,32,256,256,256,256,K_3,P_1,S_1>(fusion1_buf,outputconv_buf_1,outputconv_1,bias_outputconv_1);
				compute_conv_layer<T,32,32,256,256,256,256,K_3,P_1,S_1>(outputconv_buf_1,outputconv_buf_2,outputconv_2,bias_outputconv_2);
				fpa.RELU_Container<T>(outputconv_buf_2,outputconv_relu_buf_2);
				compute_conv_layer<T,32,1,256,256,256,256,K_3,P_1,S_1>(outputconv_buf_2,outputconv_buf_3,outputconv_3,bias_outputconv_3);
				fpa.RELU_Container<T>(outputconv_buf_3,outputconv_relu_buf_3);
				return outputconv_relu_buf_3;
			}
};
#endif /* SRC_FPT_ALGEBRA_HPP_ */
