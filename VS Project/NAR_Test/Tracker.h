#pragma once

#include <vector>
#include <cmath>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/flann.hpp"

#include "Utils.h"
#include "OpticalFlow.h"
#include "FeatureMatching.h"
#include "Filter.h"
#include "PoseFilter.h"
#include "PNP.h"
#include "Plotter.h"

///Configs
///GoodFeautures
#define MIN_MATCH_COUNT 8
#define MAX_CORNERS 100
#define QUALITY_LEVEL 0.01f
#define MIN_DISTANCE 8
#define BLOCK_SIZE 19
///Lk params
#define WIN_SIZE Size(21,21)
#define MAX_LEVEL 2
#define TRACKS_RESPAWN_RATIO 0.55f
#define BACK_THRESHOLD 1.0f
///Matching Params
#define FEATURE_COUNT 500
#define RECOGNITION_RATIO 0.5f
#define CORRECTION_RATIO 0.75f
///
#define INLIER_THRESHOLD 0.7f

class Tracker
{
private:
	cv::TermCriteria termCriteria;
	
	cv::Ptr<cv::ORB> detector;
	cv::FlannBasedMatcher matcher;
	
	std::vector<cv::KeyPoint> keypointsImg;

	cv::Mat descriptorImg;
	cv::Mat maskBounding, frameGray, framePrevGray;

	std::vector<cv::Point2f> markerCorners;
	std::vector<cv::Point2f>  matchedCorners, trackedCorners, calculatedCorners, predictedCorners;

	cv::Mat marker, markerMask;

	bool markerFound = false;
	int tracksCount;

	OpticalFlow opticalFlow;
	FeatureMatching featureMatching;
	Filter filter;
	PoseFilter poseFilter;
	PNP pnp;

	Plotter plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8;
	cv::Mat p1, p2, p3, p4, p5, p6, p7, p8, plot;

public:
	Tracker();
	Tracker(cv::Mat &marker);
	bool track(cv::Mat &frame, std::vector<cv::Point2f> &corners, bool debug);
	~Tracker();
	void reset();
	cv::Mat & getMarker();
};

