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


#define K_5 5
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
template<typename T, size_t TH, size_t TR, size_t TC> using Convk_DW = std::array<std::array<std::array<T, TC>, TR>, TH>;
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
		constexpr int Padding = padding ? (kernel_size - 1)/2 : 0;
//		static_assert(o_row == (i_row + 2 * Padding - kernel_size)/stride +1,"Conv output row dismatch!");
//		static_assert(o_col == (i_col + 2 * Padding - kernel_size)/stride +1,"Conv output col dismatch!");
		//conv operation
		
		for (int c = 0; c < o_ch; c++) {
			for (int out_i = 0; out_i < o_row; out_i++) {
				for (int out_j = 0; out_j < o_col; out_j++) {
					T sum = 0;
					for (int in_ch = 0; in_ch < i_ch; in_ch++) {
						for (int k_i = 0; k_i < kernel_size; k_i++) {
							for (int k_j = 0; k_j < kernel_size; k_j++) {
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


//0514 batchnorm
// 在Float_Point_Algebra类中添加以下成员函数
//依据onnx 不需要使用
template<typename T, size_t C, size_t H, size_t i_W>
void BATCH_NORM_2D(const Tri<T, C, H, i_W>& input,
                   const Vec<T, C>& gamma,
                   const Vec<T, C>& beta,
                   const Vec<T, C>& running_mean,
                   const Vec<T, C>& running_var,
                   Tri<T, C, H, i_W>& output,
                   T epsilon = static_cast<T>(1e-5)) {
    for (size_t c = 0; c < C; ++c) {
        // 获取通道统计量
        const T mean = running_mean[c];
        const T var = running_var[c];
        
        // 计算标准差倒数（加epsilon防除零）
        const T inv_std = static_cast<T>(1.0) / sqrt(var + epsilon);
        
        // 逐元素归一化
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < i_W; ++w) {
                const T normalized = (input[c][h][w] - mean) * inv_std;
                output[c][h][w] = normalized * gamma[c] + beta[c];
            }
        }
    }
}

//0515 深度卷积

template <typename T, size_t ch, size_t i_row, size_t i_col, size_t o_row, size_t o_col, size_t kernel_size, bool padding, int stride>
void DW_CONV_FIG(const Tri<T, ch, i_row, i_col>& FigA,
	Tri<T, ch, o_row, o_col>& FigB,
	const Convk<T, ch , ch , kernel_size, kernel_size>& kernel,
	const Bias<T, ch>& bias) {
	constexpr int Padding = padding ? (kernel_size -1)/2 : 0;
	for (int c = 0; c < ch; ++c) {
		for (int out_i = 0; out_i < o_row; ++out_i) {
			for (int out_j = 0; out_j < o_col; ++out_j) {
				T sum = 0;
				for (int k_i = 0; k_i < kernel_size; ++k_i) {
					for (int k_j = 0; k_j < kernel_size; ++k_j) {
						int in_i = out_i * stride - Padding + k_i;
						int in_j = out_j * stride - Padding + k_j;
						if (in_i >= 0 && in_i < i_row && in_j >= 0 && in_j < i_col) {
							sum += FigA[c][in_i][in_j] * kernel[c][0][k_i][k_j];
						}
						
					}
				}
				FigB[c][out_i][out_j] = sum + bias[c];
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


	/
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


		//深度卷积
		//复合深度卷积没有使用场景
		template<typename T, size_t i_ch, size_t o_ch, size_t i_row, size_t i_col, size_t o_row, size_t o_col, size_t kernel_size, bool padding, int stride>
		void depthwise_separable_conv(
			const Tri<T, i_ch, i_row, i_col>& input,
			Tri<T, o_ch, o_row, o_col>& output,
			const Convk<T, o_ch, i_ch, 1, 1>& pw_kernel,
			const Bias<T, o_ch>& pw_bias
			const Convk<T, i_ch, i_ch, kernel_size, kernel_size>& dw_kernel,
			const Bias<T, i_ch>& dw_bias
		) {
			// 计算深度卷积的中间结果
			constexpr size_t dw_o_row = (i_row + 2 * padding - kernel_size) / stride + 1;
			constexpr size_t dw_o_col = (i_col + 2 * padding - kernel_size) / stride + 1;
			Tri<T, o_ch, dw_o_row, dw_o_col> dw_result;

			// 执行逐点卷积（1x1卷积）
			compute_conv_layer<T, i_ch, o_ch, dw_o_row, dw_o_col, o_row, o_col, 1, false, 1>
				(input, dw_result, pw_kernel, pw_bias);

			
			// 执行深度卷积
			fpa.DW_CONV_FIG<T, o_ch, i_row, i_col, dw_o_row, dw_o_col, kernel_size, padding, stride>
				(dw_result, output, dw_kernel, dw_bias);

		}
		
		
		//0517
		//单独进行深度卷积
		//深度卷积使用的卷积核仍然是Convk，因为代码中就使用了[0]的输入通道
		template<typename T,size_t ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size,bool padding,int stride>
		void depthwise_conv(
							const Tri<T,ch,i_row,i_col>& input,
							Tri<T,ch,o_row,o_col>& output,
							const Convk<T, ch, ch, kernel_size, kernel_size>& dw_kernel,
							const Bias<T, ch>& dw_bias)
							{
								fpa.DW_CONV_FIG<T, ch, i_row, i_col, o_row, o_col, kernel_size, padding, stride>
								(input, output, dw_kernel, dw_bias);
							}
							
							
						
		
		
		//0517
		//最顶层封装的网络模块保持相同的模板输入

		template<typename T, size_t ch, size_t row,size_t col,size_t kernel_size,bool padding,int stride>
		void residual_block(const Tri<T,i_ch,i_row,i_col>& FigA,
								 const Convk<T,ch,ch,kernel_size,kernel_size>& kernel1,
								 const Bias<T,ch>& bias1,
								 const Convk<T,ch,ch,kernel_size,kernel_size>& kernel2,
								 const Bias<T,ch>& bias2,
								 Tri<T,ch,row,col>& FigB)
								 {
									 Tri<T,ch,row,col> result1,result2,relu_result1,relu_result2,output;
									 fpa.RELU_Container<T>(FigA,relu_result1);
									 compute_conv_layer<T, ch,ch,row,col,row,col,kernel_size,padding,stride>(relu_result1,result1,kernel1,bias1);
									 fpa.RELU_Container<T>(result1,relu_result2);
									 compute_conv_layer<T, ch,ch,row,col,row,col,kernel_size,padding,stride>(relu_result2,result2,kernel2,bias2);
									 fpa.TRI_ADD<T,ch,row,row>(result2,FigA,output);
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



			template<typename T, size_t i_ch, size_t o_ch, size_t i_row, size_t i_col, size_t o_row, size_t o_col, size_t kernel_size_3x3, size_t kernel_size_1x1, bool padding_1, bool padding_2, size_t stride_1, size_t stride_2, size_t expand>
			void efficient_block_v2(const Tri<T, i_ch, i_row, i_col>& FigA,
				//conv1
				const Convk<T, i_ch* expand, i_ch, kernel_size_1x1, kernel_size_1x1>& kernel1,
				const Bias<T, i_ch* expand>& bias1,
				//conv2
				const Convk<T, i_ch* expand, 1 , kernel_size_3x3, kernel_size_3x3>& kernel2,
				const Bias<T, i_ch* expand>& bias2,
				//conv3
				const Convk<T, i_ch, o_ch, kernel_size_1x1, kernel_size_1x1>& kernel3,
				const Bias<T, i_ch>& bias3,

				Tri<T, i_ch, o_row, o_col>& FigB)
			{
				Tri<T, o_ch, i_row, o_row> result1;
				Tri<T, i_ch, i_row, o_row>result2;
				depthwise_separable_conv<T, i_ch, o_ch, i_row, i_col, o_row, o_col, kernel_size_3x3, padding_1, stride_1>(FigA, result1, kernel1, bias1, kernel2, bias2);
				compute_conv_layer<T, o_ch, i_ch , o_row, o_col, o_row, o_col, kernel_size_1x1, padding_2, stride_1>(result1, result2, kernel3, bias3);
				fpa.TRI_ADD<T, i_ch, i_row, i_col>(FigA, result2, FigB);
			}

			//0517 此为最正统的 efficient-net v3版本
			template<typename T, size_t ch, size_t row, size_t col, size_t kernel_size_3x3, size_t kernel_size_1x1, bool padding_1, bool padding_2, size_t stride_1, size_t stride_2, size_t expand>
			void efficient_block_v3(const Tri<T, ch, row, col>& FigA,
				//conv1
				const Convk<T, ch* expand, ch, kernel_size_1x1, kernel_size_1x1>& kernel1,
				const Bias<T, ch* expand>& bias1,
				//conv2
				const Convk<T, ch* expand, ch* expand , kernel_size_3x3, kernel_size_3x3>& kernel2,
				const Bias<T, ch* expand>& bias2,
				//conv3
				const Convk<T, ch, ch, kernel_size_1x1, kernel_size_1x1>& kernel3,
				const Bias<T, ch>& bias3,

				Tri<T, ch, row, col>& FigB)
			{
				Tri<T, ch* expand, row, col> result1,result2;
				Tri<T, ch, row, col> result3;
				Tri<T, ch* expand, row, col> relu1,relu2;
				
				compute_conv_layer<T, ch, ch* expand , row, col, row, col, kernel_size_1x1, padding_2, stride_1>(FigA, result1, kernel, bias1);
				fpa.RELU_Container<T,ch* expand,row,col>(result1,relu1);
				depthwise_conv<T,ch* expand,row,col,row,col,kernel_size_3x3,padding_1,stride_1>(relu1,result2,kernel2,bias2);
				fpa.RELU_Container<T,ch* expand,row,col>(result2,relu2);
				compute_conv_layer<T, ch* expand, ch, row, col, row, col, kernel_size_1x1, padding_2, stride_1>(relu2,result3, kernel3, bias3);
				fpa.TRI_ADD<T, i_ch, i_row, i_col>(result3,FigA, FigB);
			}

			
			//0518 此为最正统的 efficient-net v5版本
			template<typename T, size_t ch, size_t row, size_t col, size_t kernel_size_5x5, size_t kernel_size_1x1, bool padding_1, bool padding_2, size_t stride_1, size_t stride_2, size_t expand>
			void efficient_block_v3(const Tri<T, ch, row, col>& FigA,
				//conv1
				const Convk<T, ch* expand, ch, kernel_size_1x1, kernel_size_1x1>& kernel1,
				const Bias<T, ch* expand>& bias1,
				//conv2
				const Convk<T, ch* expand, ch* expand , kernel_size_5x5, kernel_size_5x5>& kernel2,
				const Bias<T, ch* expand>& bias2,
				//conv3
				const Convk<T, ch, ch, kernel_size_1x1, kernel_size_1x1>& kernel3,
				const Bias<T, ch>& bias3,

				Tri<T, ch, row, col>& FigB)
			{
				Tri<T, ch* expand, row, col> result1,result2;
				Tri<T, ch, row, col> result3;
				Tri<T, ch* expand, row, col> relu1,relu2;
				
				compute_conv_layer<T, ch, ch* expand , row, col, row, col, kernel_size_1x1, padding_2, stride_1>(FigA, result1, kernel, bias1);
				fpa.RELU_Container<T,ch* expand,row,col>(result1,relu1);
				depthwise_conv<T,ch* expand,row,col,row,col,kernel_size_5x5,padding_1,stride_1>(relu1,result2,kernel2,bias2);
				fpa.RELU_Container<T,ch* expand,row,col>(result2,relu2);
				compute_conv_layer<T, ch* expand, ch, row, col, row, col, kernel_size_1x1, padding_2, stride_1>(relu2,result3, kernel3, bias3);
				fpa.TRI_ADD<T, i_ch, i_row, i_col>(result3,FigA, FigB);
			}
			
			
			
			


		//不需要了
		template<typename T,size_t i_ch,size_t i_row,size_t i_col>
		void compute_batchnorm(const Tri<T, i_ch, i_row, i_col>& input,
			const Vec<T, i_ch>& gamma,
			const Vec<T, i_ch>& beta,
			const Vec<T, i_ch>& running_mean,
			const Vec<T, i_ch>& running_var,
			Tri<T, i_ch, i_row, i_col>& output,
			T epsilon = static_cast<T>(1e-5))
		{
			fpa.BATCH_NORM_2D<T, i_ch, i_row, i_col>(input, gamma, beta, running_mean, running_var, output);
		}


		



		//命名规则：缓存类型_阶段_计算方法_编号
		//第一步  输入图像处理部分 stage1
			Convk<CF, 32, 3, 3, 3> kernel_stage1_conv_1;
			Convk<CF, 32, 32, 3, 3> kernel_stage1_convdw_2;
			Convk<CF, 24, 32, 1, 1> kernel_stage1_conv_3;
			Convk<CF, 144, 24, 1, 1> kernel_stage1_conv_4

			Bias<CF, 32> bias_stage1_conv_1;
			Bias<CF, 32> bias_stage1_convdw_2;
			Bias<CF, 24> bias_stage1_conv_3;
			Bias<CF, 144> bias_stage1_conv_4;
//此处的BUFFER编号需要修改成顺序不重复
			Tri<CF,3,256,256> buffer_input;
			Tri<CF,32,128,128> buffer_stage1_conv_1;
			Tri<CF,32,128,128> buffer_stage1_relu_1;
			Tri<CF,32,128,128> buffer_stage1_convdw_2;//dw
			Tri<CF,32,128,128> buffer_stage1_relu_2;
			Tri<CF,24,128,128> buffer_stage1_conv_3;
			Tri<CF,144,128,128> buffer_stage1_conv_4;
			Tri<CF,144,128,128> buffer_stage1_relu_3;
			
		//第二部分 尺寸缩放完成后 一个dw卷积 接 两个残差 stage2
		//命名规则：缓存类型_阶段_计算方法_编号
		//efficient-net 计算方法命名为effpw1,effdw,effpw2
			Convk<CF, 144, 144, 3, 3> kernel_stage2_convdw_1;
			Convk<CF, 32, 144, 3, 3> kernel_stage2_conv_2;
			Convk<CF, 192, 32, 1, 1> kernel_stage2_effpw1_3;
			Convk<CF, 192, 192, 3, 3> kernel_stage2_effdw_4;
			Convk<CF, 32, 192, 1, 1> kernel_stage2_effpw2_5;
			Convk<CF, 192, 32, 1, 1> kernel_stage2_effpw1_6;
			Convk<CF, 192, 192, 3, 3> kernel_stage2_effdw_7;
			Convk<CF, 32, 192, 1, 1> kernel_stage2_effpw2_8;

			Bias<CF, 144> bias_stage2_convdw_1;
			Bias<CF, 32> bias_stage2_conv_2;
			Bias<CF, 192> bias_stage2_effpw1_3;
			Bias<CF, 192> bias_stage2_effdw_4;
			Bias<CF, 32> bias_stage2_effpw2_5;
			Bias<CF, 192> bias_stage2_effpw1_6;
			Bias<CF, 192> bias_stage2_effdw_7;
			Bias<CF, 32> bias_stage2_effpw2_8;
		//efficient-net 只需要一个buffer用于结果接受 命名为eff
			Tri<CF,144,64,64>buffer_stage2_convdw_1;
			Tri<CF,144,64,64>buffer_stage2_relu_2;
			Tri<CF,32,64,64>buffer_stage2_conv_3;
			Tri<CF,32,64,64>buffer_stage2_eff_4;
			Tri<CF,32,64,64>buffer_stage2_eff_5;/////note：此为后文残差保存的第一个数据
			
		//第三部分 从第一个残差生成 到 第二个残差生成为止 stage_3
		//一个卷积 一个5x5的DW 一个卷积 两个eff-net
		//第一个dw卷积会换输入输出尺寸 要注意
		//5x5卷积 efficient-net内部使用5x5卷积
			Convk<CF,192,32,1,1> kernel_stage3_conv_1;
			Convk<CF,192,192,5,5> kernel_stage3_convdw_2;
			Convk<CF,48,192,1,1> kernel_stage3_conv_3;
			Convk<CF, 288, 48, 1, 1> kernel_stage3_effpw1_4;
			Convk<CF, 288, 288, 5, 5> kernel_stage3_effdw_5;
			Convk<CF, 48, 288, 1, 1> kernel_stage3_effpw2_6;
			Convk<CF, 288, 48, 1, 1> kernel_stage3_effpw1_7;
			Convk<CF, 288, 288, 5, 5> kernel_stage3_effdw_8;
			Convk<CF, 48, 288, 1, 1> kernel_stage3_effpw2_9;
			
			Bias<CF, 192> bias_stage3_conv_1;
			Bias<CF, 192> bias_stage3_convdw_2;
			Bias<CF, 48> bias_stage3_conv_3;
			Bias<CF, 288> bias_stage3_effpw1_4;
			Bias<CF, 288> bias_stage3_effdw_5;
			Bias<CF, 48> bias_stage3_effpw2_6;
			Bias<CF, 288> bias_stage3_effpw1_7;
			Bias<CF, 288> bias_stage3_effdw_8;
			Bias<CF, 48> bias_stage3_effpw2_9;
		
			//0518 
			Tri<CF,192,64,64>buffer_stage3_conv_1;
			Tri<CF,192,32,32>buffer_stage3_convdw_2;//5X5卷积后改变尺寸
			Tri<CF,192,32,32>buffer_stage3_relu_3;
			Tri<CF,48,32,32>buffer_stage3_conv_4;
			Tri<CF,48,32,32>buffer_stage3_eff_5;
			Tri<CF,48,32,32>buffer_stage3_eff_6;/////note：此为后文残差保存的第二个数据
		//第四部分 一个卷积 一个DW卷积 一个卷积
		//连续4个eff_v3
		//一个卷积 一个5x5的DW 一个卷积
		//连续4个eff_v5
		
			Convk<288,48,1,1>kernel_stage4_conv_1;
			Convk<288,288,3,3>kernel_stage4_convdw_2;
			Convk<96,288,1,1>kernel_stage4_conv_3;
			Convk<CF, 576, 96, 1, 1> kernel_stage4_effpw1_4;
			Convk<CF, 576, 576, 3, 3> kernel_stage4_effdw_5;
			Convk<CF, 96, 576, 1, 1> kernel_stage4_effpw2_6;	
			Convk<CF, 576, 96, 1, 1> kernel_stage4_effpw1_7;
			Convk<CF, 576, 576, 3, 3> kernel_stage4_effdw_8;
			Convk<CF, 96, 576, 1, 1> kernel_stage4_effpw2_9;	
			Convk<CF, 576, 96, 1, 1> kernel_stage4_effpw1_10;
			Convk<CF, 576, 576, 3, 3> kernel_stage4_effdw_11;
			Convk<CF, 96, 576, 1, 1> kernel_stage4_effpw2_12;	
			Convk<CF, 576, 96, 1, 1> kernel_stage4_effpw1_13;
			Convk<CF, 576, 576, 3, 3> kernel_stage4_effdw_14;
			Convk<CF, 96, 576, 1, 1> kernel_stage4_effpw2_15;	
			//eff_v5
			Convk<CF, 576, 96, 1, 1> kernel_stage4_conv_16;
			Convk<CF, 576, 576, 5, 5> kernel_stage4_convdw_17;
			Convk<CF, 136, 576, 1, 1> kernel_stage4_conv_18;		
			Convk<CF, 816, 136, 1, 1> kernel_stage4_effpw1_19;
			Convk<CF, 816, 816, 5, 5> kernel_stage4_effdw_20;
			Convk<CF, 136, 816, 1, 1> kernel_stage4_effpw2_21;
			Convk<CF, 816, 136, 1, 1> kernel_stage4_effpw1_22;
			Convk<CF, 816, 816, 5, 5> kernel_stage4_effdw_23;
			Convk<CF, 136, 816, 1, 1> kernel_stage4_effpw2_24;		
			Convk<CF, 816, 136, 1, 1> kernel_stage4_effpw1_25;
			Convk<CF, 816, 816, 5, 5> kernel_stage4_effdw_26;
			Convk<CF, 136, 816, 1, 1> kernel_stage4_effpw2_27;
			Convk<CF, 816, 136, 1, 1> kernel_stage4_effpw1_28;
			Convk<CF, 816, 816, 5, 5> kernel_stage4_effdw_29;
			Convk<CF, 136, 816, 1, 1> kernel_stage4_effpw2_30;

			Bias<CF,288> bias_stage4_conv_1;
			Bias<CF,288> bias_stage4_convdw_2;
			Bias<CF,96> bias_stage4_conv_3;
			Bias<CF,576> bias_stage4_effpw1_4;
			Bias<CF,576> bias_stage4_effdw_5;
			Bias<CF,96> bias_stage4_effpw2_6;
			Bias<CF,576> bias_stage4_effpw1_7;
			Bias<CF,576> bias_stage4_effdw_8;
			Bias<CF,96> bias_stage4_effpw2_9;
			Bias<CF,576> bias_stage4_effpw1_10;
			Bias<CF,576> bias_stage4_effdw_11;
			Bias<CF,96> bias_stage4_effpw2_12;
			Bias<CF,576> bias_stage4_effpw1_13;
			Bias<CF,576> bias_stage4_effdw_14;
			Bias<CF,96> bias_stage4_effpw2_15;
			Bias<CF,576>bias_stage4_conv_16;
			Bias<CF,576>bias_stage4_convdw_17;
			Bias<CF,136>bias_stage4_conv_18;
			Bias<CF,816> bias_stage4_effpw1_19;
			Bias<CF,816> bias_stage4_effdw_20;
			Bias<CF,136> bias_stage4_effpw2_21;
			Bias<CF,816> bias_stage4_effpw1_22;
			Bias<CF,816> bias_stage4_effdw_23;
			Bias<CF,136> bias_stage4_effpw2_24;
			Bias<CF,816> bias_stage4_effpw1_25;
			Bias<CF,816> bias_stage4_effdw_26;
			Bias<CF,136> bias_stage4_effpw2_27;
			Bias<CF,816> bias_stage4_effpw1_28;
			Bias<CF,816> bias_stage4_effdw_29;
			Bias<CF,136> bias_stage4_effpw2_30;

			Tri<CF,288,32,32>buffer_stage4_conv_1;
			Tri<CF,288,16,16>buffer_stage4_convdw_2;
			Tri<CF,288,16,16>buffer_stage4_relu_3;
			Tri<CF,96,16,16>buffer_stage4_conv_4;
			Tri<CF,96,16,16>buffer_stage4_eff_5;
			Tri<CF,96,16,16>buffer_stage4_eff_6;
			Tri<CF,96,16,16>buffer_stage4_eff_7;
			Tri<CF,96,16,16>buffer_stage4_eff_8;
			Tri<CF,576,16,16>buffer_stage4_conv_9;
			Tri<CF,576,16,16>buffer_stage4_relu_10;
			Tri<CF,576,16,16>buffer_stage4_convdw_11;
			Tri<CF,576,16,16>buffer_stage4_relu_12;
			Tri<CF,136,16,16>buffer_stage4_conv_13;
			//eff_v5
			Tri<CF,136,16,16>buffer_stage4_eff_14;
			Tri<CF,136,16,16>buffer_stage4_eff_15;
			Tri<CF,136,16,16>buffer_stage4_eff_16;
			Tri<CF,136,16,16>buffer_stage4_eff_17;////note：此为后文残差保存的第三个数据
			//stage4结束
			
			
			//stage5
			//卷积接relu
			//DW卷积 尺寸缩小 接relu
			//卷积
			//5个eff_v5
			//卷积接relu
			//dw卷积接relu
			//卷积
			//stage5结束 保留两个残差块与输入残差块的卷积不计算入stage5
			Convk<CF,816,136,1,1>kernel_stage5_conv_1;
			Convk<CF,816,816,5,5>kernel_stage5_convdw_2;
			Convk<CF,232,816,1,1>kernel_stage5_conv_3;
			//eff_v5
			Convk<CF, 1392, 232, 1, 1> kernel_stage5_effpw1_4;
			Convk<CF, 1392, 1392, 5, 5> kernel_stage5_effdw_5;
			Convk<CF, 232, 1392, 1, 1> kernel_stage5_effpw2_6;
			Convk<CF, 1392, 232, 1, 1> kernel_stage5_effpw1_7;
			Convk<CF, 1392, 1392, 5, 5> kernel_stage5_effdw_8;
			Convk<CF, 232, 1392, 1, 1> kernel_stage5_effpw2_9;
			Convk<CF, 1392, 232, 1, 1> kernel_stage5_effpw1_10;
			Convk<CF, 1392, 1392, 5, 5> kernel_stage5_effdw_11;
			Convk<CF, 232, 1392, 1, 1> kernel_stage5_effpw2_12;
			Convk<CF, 1392, 232, 1, 1> kernel_stage5_effpw1_13;
			Convk<CF, 1392, 1392, 5, 5> kernel_stage5_effdw_14;
			Convk<CF, 232, 1392, 1, 1> kernel_stage5_effpw2_15;
			Convk<CF, 1392, 232, 1, 1> kernel_stage5_effpw1_16;
			Convk<CF, 1392, 1392, 5, 5> kernel_stage5_effdw_17;
			Convk<CF, 232, 1392, 1, 1> kernel_stage5_effpw2_18;
			
			Convk<CF,1392,232,1,1>kernel_stage5_conv_19;
			Convk<CF,1392,1392,3,3>kernel_stage5_convdw_20;
			Convk<CF,384,1392,1,1>kernel_stage5_conv_21;
			
			
			
			Bias<CF,816>bias_stage5_conv_1;
			Bias<CF,816>bias_stage5_convdw_2;
			Bias<CF,232>bias_stage5_conv_3;
			
			Bias<CF,1392>bias_stage5_effpw1_4;
			Bias<CF,1392>bias_stage5_effdw_5;
			Bias<CF,232>bias_stage5_effpw2_6;
			Bias<CF,1392>bias_stage5_effpw1_7;
			Bias<CF,1392>bias_stage5_effdw_8;
			Bias<CF,232>bias_stage5_effpw2_9;
			Bias<CF,1392>bias_stage5_effpw1_10;
			Bias<CF,1392>bias_stage5_effdw_11;
			Bias<CF,232>bias_stage5_effpw2_12;
			Bias<CF,1392>bias_stage5_effpw1_13;
			Bias<CF,1392>bias_stage5_effdw_14;
			Bias<CF,232>bias_stage5_effpw2_15;
			Bias<CF,1392>bias_stage5_effpw1_16;
			Bias<CF,1392>bias_stage5_effdw_17;
			Bias<CF,232>bias_stage5_effpw2_18;
			
			Bias<CF,1392>bias_stage5_conv_19;
			Bias<CF,1392>bias_stage5_convdw_20;
			Bias<CF,384>bias_stage5_conv_21;
			
			
			Tri<CF,816,16,16>buffer_stage5_conv_1;
			Tri<CF,816,16,16>buffer_stage5_relu_2;
			Tri<CF,816,8,8>buffer_stage5_convdw_3;
			Tri<CF,816,8,8>buffer_stage5_relu_4;
			Tri<CF,232,8,8>buffer_stage5_conv_5;
			Tri<CF,232,8,8>buffer_stage5_eff_6;
			Tri<CF,232,8,8>buffer_stage5_eff_7;
			Tri<CF,232,8,8>buffer_stage5_eff_8;
			Tri<CF,232,8,8>buffer_stage5_eff_9;
			Tri<CF,232,8,8>buffer_stage5_eff_10;
			Tri<CF,1392,8,8>buffer_stage5_conv_11;
			Tri<CF,1392,8,8>buffer_stage5_relu_12;
			Tri<CF,1392,8,8>buffer_stage5_convdw_13;
			Tri<CF,1392,8,8>buffer_stage5_relu_14;
			Tri<CF,384,8,8>buffer_stage5_conv_15;
			//stage5结束
			//stage6
			//双线处理 写完一条单线再到另一条 到add结束
			Convk<CF,512,384,3,3>kernel_stage6_conv_1;
			Convk<CF,512,512,3,3>kernel_stage6_res1_2;
			Convk<CF,512,512,3,3>kernel_stage6_res2_3;
			Convk<CF,256,512,1,1>kernel_stage6_conv_4;
		
			Convk<CF,256,136,3,3>kernel_stage6_conv_5;
			Convk<CF,256,256,3,3>kernel_stage6_res1_6;
			Convk<CF,256,256,3,3>kernel_stage6_res2_7;
			
			Bias<CF,512>bias_stage6_conv_1;
			Bias<CF,512>bias_stage6_res1_2;
			Bias<CF,512>bias_stage6_res2_3;
			Bias<CF,256>bias_stage6_conv_4;
			
			Bias<CF,256>bias_stage6_conv_5;
			Bias<CF,256>bias_stage6_res1_6;
			Bias<CF,256>bias_stage6_res2_7;
			
			Tri<CF,512,8,8>buffer_stage6_conv_1;
			Tri<CF,512,8,8>buffer_stage6_res_2;
			Tri<CF,512,16,16>buffer_stage6_resize_3;
			Tri<CF,256,16,16>buffer_stage6_conv_4;
			Tri<CF,256,16,16>buffer_stage6_conv_5;
			Tri<CF,256,16,16>buffer_stage6_res_6;
			Tri<CF,256,16,16>buffer_stage6_add_7;
			
			
			
			//stage6结束 注意阶段结束后有一个add 需要单独的计算与buffer
			//stage7 
			Convk<CF,256,256,3,3>kernel_stage7_res1_1;
			Convk<CF,256,256,3,3>kernel_stage7_res2_2;
			Convk<CF,128,256,1,1>kernel_stage7_conv_3;
			
			Convk<CF,128,48,3,3>kernel_stage7_conv_4;
			Convk<CF,128,128,3,3>kernel_stage7_res1_5;
			Convk<CF,128,128,3,3>kernel_stage7_res2_6;
			
			Bias<CF,256>bias_stage7_res1_1;
			Bias<CF,256>bias_stage7_res2_2;
			Bias<CF,128>bias_stage7_conv_3;
			
			Bias<CF,128>bias_stage7_conv_4;
			Bias<CF,128>bias_stage7_res1_5;
			Bias<CF,128>bias_stage7_res1_6;
			
			Tri<CF,256,16,16>buffer_stage7_res_1;
			Tri<CF,256,32,32>buffer_stage7_resize_2;
			Tri<CF,128,32,32>buffer_stage7_conv_3;
			
			Tri<CF,128,32,32>buffer_stage7_conv_4;
			Tri<CF,128,32,32>buffer_stage7_res_5;
			Tri<CF,128,32,32>buffer_stage7_add_6;
			
			//stage7结束 注意阶段结束后有一个add 需要单独的计算与buffer
			//stage8
			Convk<CF,128,128,3,3>kernel_stage8_res1_1;
			Convk<CF,128,128,3,3>kernel_stage8_res2_2;
			Convk<CF,64,128,1,1>kernel_stage8_conv_3;
			
			Convk<CF,64,32,3,3>kernel_stage8_conv_4;
			Convk<CF,64,64,3,3>kernel_stage8_res1_5;
			Convk<CF,64,64,3,3>kernel_stage8_res2_6;
			
			Bias<CF,128>bias_stage8_res1_1;
			Bias<CF,128>bias_stage8_res2_2;
			Bias<CF,64>bias_stage8_conv_3;
			
			Bias<CF,64>bias_stage8_conv_4;
			Bias<CF,64>bias_stage8_res1_5;
			Bias<CF,64>bias_stage8_res2_6;
			
			Tri<CF,128,32,32>buffer_stage8_res_1;
			Tri<CF,128,64,64>buffer_stage8_resize_2;
			Tri<CF,64,64,64>buffer_stage8_conv_3;
			Tri<CF,64,64,64>buffer_stage8_conv_4;
			Tri<CF,64,64,64>buffer_stage8_res_5;
			Tri<CF,64,64,64>buffer_stage8_add_6;
			//stage8结束 注意阶段结束后有一个add 需要单独的计算与buffer
			//stage9
			//一个res 一个相加 一个resize 两个卷积 一个resize
			//一个卷积 一个relu 一个卷积 一个relu 一个squeeze
			Convk<CF,64,64,3,3>kernel_stage9_res1_1;
			Convk<CF,64,64,3,3>kernel_stage9_res2_2;
			Convk<CF,64,64,1,1>kernel_stage9_conv_3;
			Convk<CF,32,64,3,3>kernel_stage9_conv_4;
			Convk<CF,32,32,3,3>kernel_stage9_conv_5;
			Convk<CF,1,32,1,1>kernel_stage9_conv_6;
			
			Bias<CF,64>bias_stage9_res1_1;
			Bias<CF,64>bias_stage9_res2_2;
			Bias<CF,64>bias_stage9_conv_3;
			Bias<CF,32>bias_stage9_conv_4;
			Bias<CF,32>bias_stage9_conv_5;
			Bias<CF,1>bias_stage9_conv_6;

			Tri<CF,64,64,64>buffer_stage9_res_1;
			Tri<CF,64,128,128>buffer_stage9_resize_2;
			Tri<CF,64,128,128>buffer_stage9_conv_3;
			Tri<CF,32,128,128>buffer_stage9_conv_4;
			Tri<CF,32,256,256>buffer_stage9_resize_5;
			Tri<CF,32,256,256>buffer_stage9_conv_6;
			Tri<CF,32,256,256>buffer_stage9_relu_7;
			Tri<CF,1,256,256>buffer_stage9_conv_8;
			Tri<CF,1,256,256>buffer_stage9_relu_9;










			//每个3x3卷积模块不会减少尺寸 efficient net也存在残差模块
		//efficient-net-1 kernels
			Convk<CF, 32 * 6, 32, 1, 1> kernel_e1_1;
			Convk<CF, 32 * 6, 32 * 6, 3, 3> kernel_e1_2;
			Convk<CF, 32, 32 * 6, 1, 1> kernel_e1_3;
			Bias<CF, 192> bias_e1_1;
			Bias<CF, 192> bias_e1_2;
			Bias<CF, 192> bias_e1_3;

			//efficienCF-net-2 kernels
			Convk<CF, 32 * 6, 32, 1, 1> kernel_e2_1;
			Convk<CF, 32 * 6, 32 * 6, 3, 3> kernel_e2_2;
			Convk<CF, 48, 32 * 6, 1, 1> kernel_e2_3;
			Bias<CF, 32 * 6> bias_e2_1;
			Bias<CF, 32 * 6> bias_e2_2;
			Bias<CF, 48> bias_e2_3;
			//efficient-net-3 kernels
			Convk<CF, 48 * 6, 48, 1, 1> kernel_e3_1;
			Convk<CF, 48 * 6, 48 * 6, 3, 3> kernel_e3_2;
			Convk<CF, 136, 48 * 6, 1, 1> kernel_e3_3;
			Bias<CF, 48 * 6> bias_e3_1;
			Bias<CF, 48 * 6> bias_e3_2;
			Bias<CF, 136> bias_e3_3;
			//efficient-net-4 kernels
			Convk<CF, 136 * 6, 136, 1, 1> kernel_e4_1;
			Convk<CF, 136 * 6, 136 * 6, 3, 3> kernel_e4_2;
			Convk<CF, 384, 136 * 6, 1, 1> kernel_e4_3;
			Bias<CF, 136 * 6> bias_e4_1;
			Bias<CF, 136 * 6> bias_e4_2;
			Bias<CF, 384> bias_e4_3;

			//encoder_conv不降低特征图尺寸
			//encoder kernel
			Convk<CF, 64, 32, 3, 3> kernel_en_conv_1;
			Bias<CF, 64> bias_en_conv_1;
			Convk<CF, 128, 48, 3, 3> kernel_en_conv_2;
			Bias<CF, 128> bias_en_conv_2;
			Convk<CF, 256, 136, 3, 3> kernel_en_conv_3;
			Bias<CF, 256> bias_en_conv_3;
			Convk<CF, 512, 384, 3, 3> kernel_en_conv_4;
			Bias<CF, 512> bias_en_conv_4;

			//fusion_conv kernel
			Convk<CF, 64, 64, 3, 3> kernel_fusion1_res_1_conv1;
			Bias<CF, 64> bias_fusion1_res_1_conv1;
			Convk<CF, 64, 64, 3, 3> kernel_fusion1_res_1_conv2;
			Bias<CF, 64> bias_fusion1_res_1_conv2;
			Convk<CF, 64, 64, 3, 3> kernel_fusion1_res_2_conv1;
			Bias<CF, 64> bias_fusion1_res_2_conv1;
			Convk<CF, 64, 64, 3, 3> kernel_fusion1_res_2_conv2;
			Bias<CF, 64> bias_fusion1_res_2_conv2;
			Convk<CF, 64, 64, 1, 1> kernel_fusion1_conv3;
			Bias<CF, 64> bias_fusion1_conv3;

			Convk<CF, 128, 128, 3, 3> kernel_fusion2_res_1_conv1;
			Bias<CF, 128> bias_fusion2_res_1_conv1;
			Convk<CF, 128, 128, 3, 3> kernel_fusion2_res_1_conv2;
			Bias<CF, 128> bias_fusion2_res_1_conv2;
			Convk<CF, 128, 128, 3, 3> kernel_fusion2_res_2_conv1;
			Bias<CF, 128> bias_fusion2_res_2_conv1;
			Convk<CF, 128, 128, 3, 3> kernel_fusion2_res_2_conv2;
			Bias<CF, 128> bias_fusion2_res_2_conv2;
			Convk<CF, 64, 128, 1, 1> kernel_fusion2_conv3;
			Bias<CF, 64> bias_fusion2_conv3;

			Convk<CF, 256, 256, 3, 3> kernel_fusion3_res_1_conv1;
			Bias<CF, 256> bias_fusion3_res_1_conv1;
			Convk<CF, 256, 256, 3, 3> kernel_fusion3_res_1_conv2;
			Bias<CF, 256> bias_fusion3_res_1_conv2;
			Convk<CF, 256, 256, 3, 3> kernel_fusion3_res_2_conv1;
			Bias<CF, 256> bias_fusion3_res_2_conv1;
			Convk<CF, 256, 256, 3, 3> kernel_fusion3_res_2_conv2;
			Bias<CF, 256> bias_fusion3_res_2_conv2;
			Convk<CF, 128, 256, 1, 1> kernel_fusion3_conv3;
			Bias<CF, 128> bias_fusion3_conv3;

			Convk<CF, 512, 512, 3, 3> kernel_fusion4_res_1_conv1;
			Bias<CF, 512> bias_fusion4_res_1_conv1;
			Convk<CF, 512, 512, 3, 3> kernel_fusion4_res_1_conv2;
			Bias<CF, 512> bias_fusion4_res_1_conv2;
			Convk<CF, 512, 512, 3, 3> kernel_fusion4_res_2_conv1;
			Bias<CF, 512> bias_fusion4_res_2_conv1;
			Convk<CF, 512, 512, 3, 3> kernel_fusion4_res_2_conv2;
			Bias<CF, 512> bias_fusion4_res_2_conv2;
			Convk<CF, 256, 512, 1, 1> kernel_fusion4_conv3;
			Bias<CF, 256> bias_fusion4_conv3;

			//OutputConv kernel
			Convk<CF, 32, 64, 3, 3> outputconv_1;
			Bias<CF, 32> bias_outputconv_1;
			Convk<CF, 32, 32, 3, 3> outputconv_2;
			Bias<CF, 32> bias_outputconv_2;
			Convk<CF, 1, 32, 3, 3> outputconv_3;
			Bias<CF, 1> bias_outputconv_3;
			/////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////
			//efficient-net-buffer
			Tri<CF, 32, 128, 128> eff_buf_1;
			Tri<CF, 48, 64, 64> eff_buf_2;
			Tri<CF, 136, 32, 32> eff_buf_3;
			Tri<CF, 384, 16, 16> eff_buf_4;
			//encoder_conv_buffer
			Tri<CF, 64, 128, 128> encoder_conv_buf_1;
			Tri<CF, 128, 64, 64> encoder_conv_buf_2;
			Tri<CF, 256, 32, 32> encoder_conv_buf_3;
			Tri<CF, 512, 16, 16> encoder_conv_buf_4;
			//fusion1_buffer
			Tri<CF, 64, 256, 256> fusion1_buf;
			//fusion2_buffer
			Tri<CF, 64, 128, 128> fusion2_buf;
			//fusion3_buffer
			Tri<CF, 128, 64, 64> fusion3_buf;
			//fusion4_buffer
			Tri<CF, 256, 32, 32> fusion4_buf;
			//OutputConv
			Tri<CF, 32, 256, 256> outputconv_buf_1;
			Tri<CF, 32, 256, 256> outputconv_buf_2;
			Tri<CF, 32, 256, 256> outputconv_relu_buf_2;
			Tri<CF, 1, 256, 256> outputconv_buf_3;
			Tri<CF, 1, 256, 256> outputconv_relu_buf_3;
		template<typename T, size_t i_ch, size_t o_ch,size_t i_row,size_t i_col,size_t o_row,size_t o_col,size_t kernel_size_3x3,size_t kernel_size_1x1,bool padding_1,bool padding_2,size_t stride_1,size_t stride_2,size_t expand>
			Tri<T,1,256,256> forward()
			{
				//stage1
				compute_conv_layer<T,3,32,256,256,128,128,K_3,P_2,S_2>(buffer_input,buffer_stage1_conv_1,kernel_stage1_conv_1,bias_stage1_conv_1);
				fpa.RELU_Container<T,32,128,128>(buffer_stage1_conv_1,buffer_stage1_relu_1);
				depthwise_conv<T,32,128,128,128,128,K_3,P_1,S_1>(buffer_stage1_relu_1,buffer_stage1_conv_2,kernel_stage1_conv_2,bias_stage1_conv_2);
				fpa.RELU_Container<T,32,128,128>(buffer_stage1_conv_2,buffer_stage1_relu_2);
				compute_conv_layer<T,32,24,128,128,128,128,K_1,P_2,S_1>(buffer_stage1_relu_2,buffer_stage1_conv_3,kernel_stage1_conv_3,bias_stage1_conv_3);
				compute_conv_layer<T,24,144,128,128,128,128,K_1,P_2,S_1>(buffer_stage1_conv_3,buffer_stage1_conv_4,kernel_stage1_conv_4,bias_stage1_conv_4);
				fpa.RELU_Container<T,144,128,128>(buffer_stage1_conv_4,buffer_stage1_relu_3);
				
				//stage2
				depthwise_conv<T,144,128,128,64,64,K_3,P_1,S_2>(buffer_stage1_relu_3,buffer_stage2_convdw_1,kernel_stage2_convdw_1,bias_stage2_convdw_1);
				fpa.RELU_Container<T,144,64,64>(buffer_stage2_convdw_1,buffer_stage2_relu_2);
				compute_conv_layer<T,144,32,64,64,64,64,K_1,P_2,S_1>(buffer_stage2_relu_2,buffer_stage2_conv_3,kernel_stage2_conv_2,bias_stage2_conv_2);
				efficient_block_v3<T,32,64,64,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage2_conv_3,kernel_stage2_effpw1_3,bias_stage2_effpw1_3,kernel_stage2_effdw_4,bias_stage2_effdw_4,kernel_stage2_effpw2_5,bias_stage2_effpw2_5,buffer_stage2_eff_4);
				efficient_block_v3<T,32,64,64,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage2_eff_4,kernel_stage2_effpw1_6,bias_stage2_effpw1_6,kernel_stage2_effdw_7,bias_stage2_effdw_7,kernel_stage2_effpw2_8,bias_stage2_effpw2_8,buffer_stage2_eff_5);
				
				//stage3
				compute_conv_layer<T,32,192,64,64,64,64,K_1,P_2,S_1>(buffer_stage2_eff_5,buffer_stage3_conv_1,kernel_stage3_conv_1,bias_stage3_conv_1);
				depthwise_conv<T,192,64,64,32,32,K_5,P_1,S_2>(buffer_stage3_conv_1,buffer_stage3_convdw_2,kernel_stage3_convdw_2,bias_stage3_convdw_2);
				fpa.RELU_Container<T,192,32,32>(buffer_stage3_convdw_2,buffer_stage3_relu_3);
				compute_conv_layer<T,192,48,32,32,32,32,K_1,P_2,S_1>(buffer_stage3_relu_3,buffer_stage3_conv_4,kernel_stage3_conv_3,bias_stage3_conv_3);
				efficient_block_v5<T,48,32,32,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage3_conv_4,kernel_stage3_effpw1_4,bias_stage3_effpw1_4,kernel_stage3_effdw_5,bias_stage3_effdw_5,kernel_stage3_effpw2_6,bias_stage3_effpw2_6,buffer_stage3_eff_5);
				efficient_block_v5<T,48,32,32,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage3_eff_5,kernel_stage3_effpw1_7,bias_stage3_effpw1_7,kernel_stage3_effdw_8,bias_stage3_effdw_8,kernel_stage3_effpw2_9,bias_stage3_effpw2_9,buffer_stage3_eff_6);
				
				//stage4
				compute_conv_layer<T,48,288,32,32,32,32,K_1,P_2,S_1>(buffer_stage3_eff_6,buffer_stage4_conv_1,kernel_stage4_conv_1,bias_stage4_conv_1);
				depthwise_conv<T,288,32,32,16,16,K_3,P_2,S_2>(buffer_stage4_conv_1,buffer_stage4_convdw_2,kernel_stage4_convdw_2,bias_stage4_convdw_2);
				fpa.RELU_Container<T,288,16,16>(buffer_stage4_convdw_2,buffer_stage4_relu_3);
				compute_conv_layer<T,288,96,16,16,16,16,K_1,P_2,S_1>(buffer_stage4_relu_3,buffer_stage4_conv_4,kernel_stage4_conv_3,bias_stage4_conv_3);
				efficient_block_v3<T,96,16,16,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_conv_4,kernel_stage4_effpw1_4,bias_stage4_effpw1_4,kernel_stage4_effdw_5,bias_stage4_effdw_5,kernel_stage4_effpw2_6,bias_stage4_effpw2_6,buffer_stage4_eff_5);
				efficient_block_v3<T,96,16,16,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_eff_5,kernel_stage4_effpw1_7,bias_stage4_effpw1_7,kernel_stage4_effdw_8,bias_stage4_effdw_8,kernel_stage4_effpw2_9,bias_stage4_effpw2_9,buffer_stage4_eff_6);
				efficient_block_v3<T,96,16,16,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_eff_6,kernel_stage4_effpw1_10,bias_stage4_effpw1_10,kernel_stage4_effdw_11,bias_stage4_effdw_11,kernel_stage4_effpw2_12,bias_stage4_effpw2_12,buffer_stage4_eff_7);
				efficient_block_v3<T,96,16,16,K_3,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_eff_7,kernel_stage4_effpw1_13,bias_stage4_effpw1_13,kernel_stage4_effdw_14,bias_stage4_effdw_14,kernel_stage4_effpw2_15,bias_stage4_effpw2_15,buffer_stage4_eff_8);
				compute_conv_layer<T,96,576,16,16,16,16,K_1,P_2,S_1>(buffer_stage4_eff_8,buffer_stage4_conv_9,kernel_stage4_conv_16,bias_stage4_conv_16);
				fpa.RELU_Container<T,576,16,16>(buffer_stage4_conv_9,buffer_stage4_relu_10);
				depthwise_conv<T,576,16,16,16,16,K_5,P_1,S_1>(buffer_stage4_relu_10,buffer_stage4_convdw_11,kernel_stage4_convdw_17,bias_stage4_convdw_17);
				fpa.RELU_Container<T,576,16,16>(buffer_stage4_convdw_11,buffer_stage4_relu_12);
				compute_conv_layer<T,576,136,16,16,16,16,K_1,P_2,S_1>(buffer_stage4_relu_12,buffer_stage4_conv_13,kernel_stage4_conv_18,bias_stage4_conv_18);
				efficient_block_v5<T,136,16,16,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_conv_13,kernel_stage4_effpw1_19,bias_stage4_effpw1_19,kernel_stage4_effdw_20,bias_stage4_effdw_20,kernel_stage4_effpw2_21,bias_stage4_effpw2_21,buffer_stage4_eff_14);
				efficient_block_v5<T,136,16,16,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_eff_14,kernel_stage4_effpw1_22,bias_stage4_effpw1_22,kernel_stage4_effdw_23,bias_stage4_effdw_23,kernel_stage4_effpw2_24,bias_stage4_effpw2_24,buffer_stage4_eff_15);
				efficient_block_v5<T,136,16,16,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_eff_15,kernel_stage4_effpw1_25,bias_stage4_effpw1_25,kernel_stage4_effdw_26,bias_stage4_effdw_26,kernel_stage4_effpw2_27,bias_stage4_effpw2_27,buffer_stage4_eff_16);
				efficient_block_v5<T,136,16,16,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage4_conv_16,kernel_stage4_effpw1_28,bias_stage4_effpw1_28,kernel_stage4_effdw_29,bias_stage4_effdw_29,kernel_stage4_effpw2_30,bias_stage4_effpw2_30,buffer_stage4_eff_17);
				
				//stage5
				compute_conv_layer<T,136,816,16,16,16,16,K_1,P_2,S_1>(buffer_stage4_eff_17,buffer_stage5_conv_1,kernel_stage5_conv_1,bias_stage5_conv_1);
				fpa.RELU_Container<T,816,16,16>(buffer_stage5_conv_1,buffer_stage5_relu_2);
				depthwise_conv<T,816,16,16,8,8,K_5,P_2,S_2>(buffer_stage5_relu_2,buffer_stage5_convdw_3,kernel_stage5_convdw_2,bias_stage5_convdw_2);
				fpa.RELU_Container<T,816,8,8>(buffer_stage5_convdw_3,buffer_stage5_relu_4);
				compute_conv_layer<T,816,232,8,8,8,8,K_1,P_2,S_1>(buffer_stage5_relu_4,buffer_stage5_conv_5,kernel_stage5_conv_3,bias_stage5_conv_3);
				efficient_block_v5<T,232,8,8,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage5_conv_5,kernel_stage5_effpw1_4,bias_stage5_effpw1_4,kernel_stage5_effdw_5,bias_stage5_effdw_5,kernel_stage5_effpw2_6,bias_stage5_effpw2_6,buffer_stage5_eff_6);
				efficient_block_v5<T,232,8,8,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage5_eff_6,kernel_stage5_effpw1_7,bias_stage5_effpw1_7,kernel_stage5_effdw_8,bias_stage5_effdw_8,kernel_stage5_effpw2_9,bias_stage5_effpw2_9,buffer_stage5_eff_7);
				efficient_block_v5<T,232,8,8,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage5_eff_7,kernel_stage5_effpw1_10,bias_stage5_effpw1_10,kernel_stage5_effdw_11,bias_stage5_effdw_11,kernel_stage5_effpw2_12,bias_stage5_effpw2_12,buffer_stage5_eff_8);
				efficient_block_v5<T,232,8,8,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage5_eff_8,kernel_stage5_effpw1_13,bias_stage5_effpw1_13,kernel_stage5_effdw_14,bias_stage5_effdw_14,kernel_stage5_effpw2_15,bias_stage5_effpw2_15,buffer_stage5_eff_9);
				efficient_block_v5<T,232,8,8,K_5,K_1,P_1,P_2,S_1,S_2,6>(buffer_stage5_eff_9,kernel_stage5_effpw1_16,bias_stage5_effpw1_16,kernel_stage5_effdw_17,bias_stage5_effdw_17,kernel_stage5_effpw2_18,bias_stage5_effpw2_18,buffer_stage5_eff_10);
				compute_conv_layer<T,232,1392,8,8,8,8,K_1,P_2,S_1>(buffer_stage5_eff_10,buffer_stage5_conv_11,kernel_stage5_conv_19,bias_stage5_conv_19);
				fpa.RELU_Container<T,1392,8,8>(buffer_stage5_conv_11,buffer_stage5_relu_12);
				depthwise_conv<T,1392,8,8,8,8,K_3,P_1,S_1>(buffer_stage5_relu_12,buffer_stage5_convdw_13,kernel_stage5_convdw_20,bias_stage5_convdw_20);
				fpa.RELU_Container<T,1392,8,8>(buffer_stage5_convdw_13,buffer_stage5_relu_14);
				compute_conv_layer<T,384,1392,8,8,8,8,K_1,P_2,S_1>(buffer_stage5_relu_14,buffer_stage5_conv_15,kernel_stage5_conv_21,bias_stage5_conv_21);
				
				//stage5结束
				//stage6开始
				compute_conv_layer<T,384,512,8,8,8,8,K_3,P_1,S_1>(buffer_stage5_conv_15,buffer_stage6_conv_1,kernel_stage6_conv_1,bias_stage6_conv_1);
				residual_block<T,512,8,8,K_3,P_1,S_1>(buffer_stage6_conv_1,kernel_stage6_res1_2,bias_stage6_res1_2,kernel_stage6_res2_3,bias_stage6_res2_3,buffer_stage6_res_2);
				fpa.BILINEAR_INTERPOLATION_TRI<T,512,8,8,16,16>(buffer_stage6_res_2,buffer_stage6_resize_3);
				compute_conv_layer<T,512,256,16,16,16,16,K_1,P_2,S_1>(buffer_stage6_resize_3,buffer_stage6_conv_4,kernel_stage6_conv_4,bias_stage6_conv_4);
				compute_conv_layer<T,136,256,16,16,16,16,K_3,P_1,S_1>(buffer_stage4_eff_17,buffer_stage6_conv_5,kernel_stage6_conv_5,bias_stage6_conv_5);
				residual_block<T,256,16,16,K_3,P_1,S_1>(buffer_stage6_conv_5,kernel_stage6_res1_6,bias_stage6_res1_6,kernel_stage6_res2_7,bias_stage6_res2_7,buffer_stage6_res_6);
				fpa.TRI_ADD<T,256,16,16>(buffer_stage6_conv_4,buffer_stage6_res_6,buffer_stage6_add_7);
				
				
				//stage7
				residual_block<T,256,16,16,K_3,P_1,S_1>(buffer_stage6_add_7,kernel_stage7_res1_1,bias_stage7_res1_1,kernel_stage7_res2_2,bias_stage7_res2_2,buffer_stage7_res_1);
				fpa.BILINEAR_INTERPOLATION_TRI<T,256,16,16,32,32>(buffer_stage7_res_1,buffer_stage7_resize_2);
				compute_conv_layer<T,256,128,32,32,32,32,K_1,P_2,S_1>(buffer_stage7_resize_2,buffer_stage7_conv_3,kernel_stage7_conv_3,bias_stage7_conv_3);
				compute_conv_layer<T,48,128,32,32,32,32,K_3,P_1,S_1>(buffer_stage3_eff_6,buffer_stage7_conv_4,kernel_stage7_conv_4,bias_stage7_conv_4);
				residual_block<T,128,32,32,K_3,P_1,S_1>(buffer_stage7_conv_4,kernel_stage7_res1_5,bias_stage7_res1_5,kernel_stage7_res2_6,bias_stage7_res1_6,buffer_stage7_res_5);
				fpa.TRI_ADD<T,128,32,32>(buffer_stage7_conv_3,buffer_stage7_res_5,buffer_stage7_add_6);
				
				
				//stage8
				residual_block<T,128,32,32,K_3,P_1,S_1>(buffer_stage7_add_6,kernel_stage8_res1_1,bias_stage8_res1_1,kernel_stage8_res2_2,bias_stage8_res2_2,buffer_stage8_res_1);
				fpa.BILINEAR_INTERPOLATION_TRI<T,128,32,32,64,64>(buffer_stage8_res_1,buffer_stage8_resize_2);
				compute_conv_layer<T,128,64,64,64,64,64,K_1,P_2,S_1>(buffer_stage8_resize_2,buffer_stage8_conv_3,kernel_stage8_conv_3,bias_stage8_conv_3);
				compute_conv_layer<T,32,64,64,64,64,64,K_3,P_1,S_1>(buffer_stage2_eff_5,buffer_stage8_conv_4,kernel_stage8_conv_4,bias_stage8_conv_4);
				residual_block<T,64,64,64,K_3,P_1,S_1>(buffer_stage8_conv_4,kernel_stage8_res1_5,bias_stage8_res1_5,kernel_stage8_res2_6,bias_stage8_res2_6,buffer_stage8_res_5);
				fpa.TRI_ADD<T,64,64,64>(buffer_stage8_conv_3,buffer_stage8_res_5,buffer_stage8_add_6);
				
				
				//stage9
				residual_block<T,64,64,64,K_3,P_1,S_1>(buffer_stage8_add_6,kernel_stage9_res1_1,bias_stage9_res1_1,kernel_stage9_res2_2,bias_stage9_res2_2,buffer_stage9_res_1);
				fpa.BILINEAR_INTERPOLATION_TRI<T,64,64,64,128,128>(buffer_stage9_res_1,buffer_stage9_resize_2);
				compute_conv_layer<T,64,64,128,128,128,128,K_1,P_2,S_1>(buffer_stage9_resize_2,buffer_stage9_conv_3,kernel_stage9_conv_3,bias_stage9_conv_3);
				compute_conv_layer<T,64,32,128,128,128,128,K_3,P_1,S_1>(buffer_stage9_conv_3,buffer_stage9_conv_4,kernel_stage9_conv_4,bias_stage9_conv_4);
				fpa.BILINEAR_INTERPOLATION_TRI<T,32,128,128,256,256>(buffer_stage9_conv_4,buffer_stage9_resize_5);
				compute_conv_layer<T,32,32,256,256,256,256,K_3,P_1,S_1>(buffer_stage9_resize_5,buffer_stage9_conv_6,kernel_stage9_conv_5,bias_stage9_conv_5);
				fpa.RELU_Container<T,32,256,256>(buffer_stage9_conv_6,buffer_stage9_relu_7);
				compute_conv_layer<T,32,1,256,256,256,256,K_1,P_2,S_1>(buffer_stage9_relu_7,buffer_stage9_conv_8,kernel_stage9_conv_6,bias_stage9_conv_6);
				fpa.RELU_Container<T,1,256,256>(buffer_stage9_conv_8,buffer_stage9_relu_9);
				
			}
};
#endif /* SRC_FPT_ALGEBRA_HPP_ */
