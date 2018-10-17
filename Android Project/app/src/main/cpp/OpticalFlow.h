#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"


#include <vector>
#include <cmath>

#include "Utils.h"

class OpticalFlow
{
private:
	///OpticalFLow configs
	cv::Size winSize;
	int maxLevel;
	float tracksRespawnRatio;
	cv::TermCriteria termCriteria;

	///GoodFeaturesToTrack configs
	float backThreshold;
	int maxCorners;
	float qualityLevel;
	int minDistance;
	int blockSize;

	cv::Mat markerMask;

	std::vector<unsigned char> trackStatus;
	std::vector<float> err;
	std::vector<cv::Point2f> tracksNew, tracksRev;

	cv::Mat homography;
	std::vector<cv::Mat> nextPyr, prevPyr;
public:
	std::vector<cv::Point2f> tracks;

	OpticalFlow();
	OpticalFlow(cv::Size , int , float , cv::TermCriteria , float, int , float , int , int, cv::Mat);
	~OpticalFlow();
	bool track(std::vector<cv::Point2f> &, cv::Mat&, cv::Mat&, int&);
	void reset();
};



