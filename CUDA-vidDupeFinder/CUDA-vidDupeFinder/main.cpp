//#include<opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>


#include "Header.h"

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
	//playVid();
	//createFrames();
	//ffmpegtest();
	//getVidTime();
	//pullMetadata();
	//MIVidTime();
	//printSplits(5);
	char buff[FILENAME_MAX]; //create string buffer to hold path
	GetCurrentDir(buff, FILENAME_MAX);
	string current_working_dir(buff);
	cout << current_working_dir;
	findSimilarVideos("testVids/iglesias_test_vid_orig.mp4", "./testvids/", .05);
	getFramesAtSplits("testVids/iglesias_test_vid_orig.mp4", 2000, 5);
	return 0;
}