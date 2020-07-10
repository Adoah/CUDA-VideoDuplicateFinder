#include "..\..\..\..\..\..\Program Files\mediainfoDLL\Developers\Source\MediaInfoDLL\MediaInfoDLL.h"
#define MediaInfoNameSpace MediaInfoDLL;
#include "Header.h"
#include <iostream>

using namespace MediaInfoNameSpace;

int mediaInfoTest() //returns time of video in ms
{
    //Information about MediaInfo
    MediaInfo MI;
    std::string To_Display = MI.Option(__T("Info_Version"), __T("0.7.13;MediaInfoDLL_Example_MSVC;0.7.13"));

    //An example of how to use the library
    To_Display += __T("\r\n\r\nOpen\r\n");
    MI.Open(__T("iglesias_test_vid_orig.mp4"));

    To_Display += __T("\r\n\r\nInform with Complete=true\r\n");
    MI.Option(__T("Complete"), __T("1"));

    To_Display += __T("\r\n\r\nGet with Stream=Video and Parameter=\"Duration\"\r\n");
    To_Display += MI.Get(Stream_Video, 0, __T("Duration"), Info_Text, Info_Name); //money maker line
    int timeinmillis = std::stoi(MI.Get(Stream_Video, 0, __T("Duration")));
    //it gets the length of the video in ms

    To_Display += __T("\r\n\r\nClose\r\n");
    MI.Close();

    std::cout << To_Display;
    std::cout << timeinmillis;

    return timeinmillis;
}

