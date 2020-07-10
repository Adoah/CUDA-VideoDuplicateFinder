#pragma once

#define FRONTREMOVE 30
#define COMPARE_TIME 2000
#define SPLIT_COUNT 5


int playVid();
int createFrames();

using namespace System;
using namespace System::Collections;
using namespace std;
#include <vector>
#include <opencv2/opencv.hpp>


int MIVidTime(string videoName);
int ffmpegVidTime();
int printSplits(int N);
ArrayList^ splitTimes(string videoName, int N);
int MIVidFPS(string videoName);
std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>  getFramesAtSplits(string videoName, int amtOfFramesInMS, int amountOfSplits);
std::vector<cv::Mat, std::allocator<cv::Mat>> createFramesInBounds(string videoName, int startFrame, int duration, int splitNum, std::vector<cv::Mat, std::allocator<cv::Mat>> writeLocation);
std::vector<std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>> findSimilarVideos(string sourceVideo, System::String^ duplicateDirectory, double error);

void MarshalString(System::String^ s, string& os);
//std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>> buildFramesOnVideos();
std::vector<std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>> buildFramesOnVideos(string sourceVideo, vector<string> similarVideos);

