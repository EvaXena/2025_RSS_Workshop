
#include "fpt_algebra.hpp"
#include <iostream>
#include <exception>  // 支持标准异常类
#include <stdexcept>  // 支持运行时错误异常

int main()
{


	Tri<CF,3,256,256> input;

	for(int c = 0; c < 3; ++c)
	{
		for(int i=0;i < 256;++i)
		{
			for(int j=0;j < 256;++j)
			{
				input[c][i][j] = ((i/16 + j/16)%2) ? CF(0.2):CF(0.8);
			}
		}
	}

	Net net;


	auto output = net.forward<CF,3,1,256,256,256,256,3,1,true,false,2,1,6>();


	return 0;
}
