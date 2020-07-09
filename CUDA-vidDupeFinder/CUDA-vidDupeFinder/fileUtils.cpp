#include <Windows.h>
#include <iostream>
#include <sstream>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include "..\..\..\..\..\..\Program Files\mediainfoDLL\Developers\Source\MediaInfoDLL\MediaInfoDLL.h"
#define MediaInfoNameSpace MediaInfoDLL;
#include "Header.h"
#include <iostream>

#include "Header.h"
#include "ffmpegcpp.h"

#define FRONTREMOVE 30

using namespace std;
using namespace ffmpegcpp;

int ffmpegtest()
{
	// This example will print info about a video.
	try
	{
		// Load a video from a container and read its info.
		const char* fileName = "iglesias_test_vid_orig.mp4";
		Demuxer* demuxer = new Demuxer(fileName);
		ContainerInfo info = demuxer->GetInfo();

		// Print the data similar to ffmpeg.exe.
		cout << "Input " << info.format->name << " from '" << fileName << "'" << endl;

		cout << "Video streams:" << endl;
		for (int i = 0; i < info.videoStreams.size(); ++i)
		{
			VideoStreamInfo stream = info.videoStreams[i];
			cout << "Stream #" << (i + 1)
				<< ": codec " << stream.codec->name
				<< ", pixel format " << stream.formatName
				<< ", resolution " << stream.width << "x" << stream.height
				<< ", bit rate " << stream.bitRate << "kb/s"
				<< ", fps " << ((float)stream.frameRate.num / (float)stream.frameRate.den)
				<< ", time base " << stream.timeBase.num << "/" << stream.timeBase.den
				<< ", " << demuxer->GetFrameCount(stream.id) << " frames"			
				<< endl;
		}

		cout << "Audio streams:" << endl;
		for (int i = 0; i < info.audioStreams.size(); ++i)
		{
			AudioStreamInfo stream = info.audioStreams[i];
			cout << "Stream #" << (i + 1)
				<< ": codec " << stream.codec->name
				<< ", channel layout " << stream.channelLayoutName
				<< ", channels " << stream.channels
				<< ", bit rate " << stream.bitRate << "kb/s"
				<< ", sample rate " << stream.sampleRate
				<< ", time base " << stream.timeBase.num << "/" << stream.timeBase.den
				<< ", " << demuxer->GetFrameCount(stream.id) << " frames"
				<< endl;
		}
	}
	catch (FFmpegException e)
	{
		cerr << "Exception caught!" << endl;
		cerr << e.what() << endl;
		throw e;
	}

	cout << "Encoding complete!" << endl;
	cout << "Press any key to continue..." << endl;

	getchar();
}

int ffmpegVidTime()
{
	const char* fileName = "iglesias_test_vid_orig.mp4";
	Demuxer* demuxer = new Demuxer(fileName);
	ContainerInfo info = demuxer->GetInfo();

	VideoStreamInfo stream = info.videoStreams[0];

	cout << demuxer->GetFrameCount(stream.id) << " frames";

	return demuxer->GetFrameCount(stream.id);
}

using namespace MediaInfoNameSpace;
int MIVidTime(string videoName) //returns time of video in ms using media info
{
    //Information about MediaInfo
    MediaInfo MI;
   
    //An example of how to use the library
    MI.Open(__T(videoName));

    int timeinmillis = std::stoi(MI.Get(Stream_Video, 0, __T("Duration")));
    //it gets the length of the video in ms

    MI.Close();

    std::cout << timeinmillis << endl;

    return timeinmillis;
}

int MIVidFPS(string videoName)
{
	MediaInfo MI;

	MI.Open(__T(videoName));

	string To_Display = __T("\r\n\r\nGet with Stream=Video and Parameter=\"framerate\"\r\n");
	To_Display += MI.Get(Stream_Video, 0, __T("FrameRate"), Info_Text, Info_Name); //money maker line
	int fps = std::stoi(MI.Get(Stream_Video, 0, __T("FrameRate")));

	std::cout << To_Display;
	std::cout << fps << "fps" << endl;

	return fps;
}

using namespace System;
using namespace System::Collections;
//create the timing gaps for N amount of gaps
ArrayList^ splitTimes(string videoName, int N)
{
	int vidtime = MIVidTime(videoName) - (FRONTREMOVE * 2000); //removing FRONTREMOVE from the front and back of the footage

	int splitDuration = vidtime / N;

	ArrayList^ splits = gcnew ArrayList;
	
	for (int i = 1; i <= N; i++)
	{
		splits->Add((i * splitDuration) + (FRONTREMOVE * 1000)); //re-adding FRONTREMOVE seconds of footage (attempting to remove the first and last FRONTREMOVE seconds of the video from the computation)
	}

	return splits;
}

//prints out split timings
int printSplits(string videoName, int N) 
{
	ArrayList^ splits = splitTimes(videoName, N);

	IEnumerator^ enumerator = splits->GetEnumerator();
	while (enumerator->MoveNext())
	{
		//Cast the object into its corresponding type
		Object^ object = safe_cast<Object^>(enumerator->Current);
		//Print the type followed by the value of the variable
		Console::WriteLine("Type: " + object->GetType() + " Value: " + object);
	}
	return 1;
}

int buildFramesOnVideos()
{

	return 1;
}

int findSimilarVideos(string sourceVideo, System::String^ duplicateDirectory, double error)
{
//scan directories for video files with similar lengths, accounting for the %error
	//get the target time	
	int sourceVideoTime = MIVidTime(sourceVideo);
	//upper and lower bounds for target time
	int lowerbound = sourceVideoTime * (1 - error);
	int upperbound = sourceVideoTime * (1 + error);

	cli::array<System::String^, 1>^ files = System::IO::Directory::GetFiles(duplicateDirectory, "*.mp4", System::IO::SearchOption::AllDirectories);

	vector<string> similarVideos;

	IEnumerator^ enumerator = files->GetEnumerator();
	while (enumerator->MoveNext())
	{
		//Cast the object into its corresponding type
		Object^ object = safe_cast<Object^>(enumerator->Current);
		//Print the type followed by the value of the variable
		Console::WriteLine("Type: " + object->GetType() + " Value: " + object);
		Console::WriteLine(object->ToString());
		System::String^ sysString = object->ToString();
		string targetVideo = "test";
		MarshalString(sysString, targetVideo);
		cout << targetVideo << endl;
		int targetVideoTime = MIVidTime(targetVideo);
		if (targetVideoTime > lowerbound && targetVideoTime < upperbound)
		{
			//add targetVideo to arraylist/vector
			similarVideos.push_back(targetVideo);
		}
		//if not, loop to next in array
	}


	//build arraylist with those similar video files
	//send that arraylist to buildframesonvideos()
	return 1;
}

//this is necessary to convert from System::String^ to a std::string
void MarshalString(System::String^ s, string& os) 
{
	using namespace Runtime::InteropServices;
	const char* chars =
		(const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	os = chars;
	Marshal::FreeHGlobal(IntPtr((void*)chars));
}
