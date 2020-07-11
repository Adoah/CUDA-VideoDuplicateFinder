#include<opencv2/opencv.hpp>
#include<iostream>
#include "Header.h"
using namespace std;
using namespace cv;
int playVid()
{
	VideoCapture capture("testVids/iglesias_test_vid_orig.mp4");

    capture.set(cv::CAP_PROP_POS_FRAMES, 4000);
    
    if (!capture.isOpened())
		throw "error when reading video stream";


    int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, 1); //resizable window;
    Mat frame;

    for (;;) {
        capture >> frame;
        if (frame.empty())
            break;

        imshow(window_name, frame);
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;
        case ' ': //Save an image
            sprintf(filename, "filename%.3d.jpg", n++); //this is just to manage the filenames
            imwrite(filename, frame); //writes the frame to image
            cout << "Saved " << filename << endl; //says that it was saved
            break;
        default:
            break;
        }
    }

    // When everything done, release the video capture object
    capture.release();

    // Closes all the frames
    destroyAllWindows();
	return 0;
}

//will now build frames from video file
int createFrames()
{
    VideoCapture capture("testVids/iglesias_test_vid_orig.mp4");

    if (!capture.isOpened())
        throw "error when reading video stream";


    int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, 1); //resizable window;
    Mat frame;

    for (;;) {
        capture >> frame;
        if (frame.empty())
            break;

        //imshow(window_name, frame);
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        sprintf(filename, "filename%.3d.jpg", n++); //this is just to manage the filenames
        imwrite(filename, frame); //writes the frame to image
        //cout << "Saved " << filename << endl; //says that it was saved


        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;
        default:
            break;
        }
    }

    // When everything done, release the video capture object
    capture.release();

    // Closes all the frames
    //destroyAllWindows();
    return 0;
}

using namespace System;
using namespace System::Collections;
//takes the video name, frame split duration and amount of splits and creates vector object with frames
std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>  getFramesAtSplits(string videoName, int amtOfFramesInMS, int amountOfSplits)
{
    ArrayList^ splits = splitTimes(videoName, amountOfSplits);
    int amtOfFrames = (amtOfFramesInMS / 1000) * (int)MIVidFPS(videoName);
    int splitNum = 0;
    IEnumerator^ enumerator = splits->GetEnumerator();
    std::vector<std::vector<Mat>> videoFrames(amountOfSplits, std::vector<Mat>(amtOfFrames));
    while (enumerator->MoveNext())
    {
        //Cast the object into its corresponding type
        Object^ object = safe_cast<Object^>(enumerator->Current);
        //Print the type followed by the value of the variable
        Console::WriteLine("Type: " + object->GetType() + " Value: " + object);
        int frameTarget = ((int)object / 1000) * (int)MIVidFPS(videoName);
        videoFrames.at(splitNum) = createFramesInBounds(videoName, frameTarget, amtOfFrames, splitNum, videoFrames.at(splitNum));
        splitNum++;
        cout << splitNum << "splitnum" << endl;
    }

    return videoFrames;
}


std::vector<cv::Mat, std::allocator<cv::Mat>> createFramesInBounds(string videoName, int startFrame, int duration, int splitNum, std::vector<cv::Mat, std::allocator<cv::Mat>> writeLocation)
{
    VideoCapture capture(videoName);
    
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    if (!capture.isOpened())
        throw "error when reading video stream";
    std::cout << startFrame << " startframe" << endl;
    std::cout << duration << " duration" << endl;

    //int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "q or esc to quit" << endl;
    namedWindow(window_name, 1); //resizable window;
    Mat frame;

    cout << splitNum << " splitNum" << endl;

    for (int n = 0;n < duration; n++) {
        capture >> frame;
        if (frame.empty())
            break;

        //imshow(window_name, frame);
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        sprintf(filename, "cappedFrame/split%d.frame%.3d.jpg", splitNum, n); //this is just to manage the filenames
        //imwrite(filename, frame); //writes the frame to image
        //cout << "Saved " << filename << endl; //says that it was saved
        writeLocation.at(n) = frame;


        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return writeLocation;
        default:
            break;
        }
    }

    // When everything done, release the video capture object
    capture.release();

    // Closes all the frames
    //destroyAllWindows();
    return writeLocation;
}

//show frame from vector set