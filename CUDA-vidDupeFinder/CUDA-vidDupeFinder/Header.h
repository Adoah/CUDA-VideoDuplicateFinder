#pragma once
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
int findSimilarVideos(string sourceVideo, System::String^ duplicateDirectory, double error);

void MarshalString(System::String^ s, string& os);

