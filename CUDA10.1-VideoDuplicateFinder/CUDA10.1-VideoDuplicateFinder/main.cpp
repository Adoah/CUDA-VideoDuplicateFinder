//#include<opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>


#include "Header.h"
#include "CUDA-header.h"

#include <direct.h>
#define GetCurrentDir _getcwd
#include <stdio.h>

using namespace std;

//using namespace cv;
int main(int argc, char** argv)
{
	//Mat img = imread("lena.jpg");
	//namedWindow("image", WINDOW_NORMAL);
	//imshow("image", img);
	//waitKey(0);
	cout << "Hello world!";
	cout << "\nThis is coming from the c++ file!";
	char buff[FILENAME_MAX]; //create string buffer to hold path
	GetCurrentDir(buff, FILENAME_MAX);
	string current_working_dir(buff);
	cout << current_working_dir << endl;
	cout << "finding similar videos!" << endl;
	std::vector<std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>> builtFrames = findSimilarVideos("testVids/iglesias_test_vid_orig.mp4", "./testvids/", .05);
	vectorBreaker(builtFrames);
	return 0;
}