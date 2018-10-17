#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <cmath>

#include "Utils.h"

class FeatureMatching
{
private:

	std::vector<cv::KeyPoint> keypointsDetected;
	cv::Mat descriptorDetected;
	std::vector<cv::Point2f> markerCorners;
	std::vector<cv::DMatch> goodMatches;
	std::vector<std::vector<cv::DMatch>> matches;

	cv::Mat homography;

	std::vector<cv::KeyPoint> matchedKeypoints;
	std::vector<cv::Point2f> obj, scene;

	cv::Ptr<cv::ORB> detector;
	cv::FlannBasedMatcher matcher;

	std::vector<cv::KeyPoint> keypointsImg;
	cv::Mat descriptorImg;

	int minMatchCount;
	float recognitionRatio, correctionRatio;
	int maxCorners;
	float qualityLevel;
	int minDistance;
	int blockSize;
	float tracksRespawnRatio;
	float inlierThershold;
	cv::Mat markerMask;

	void reset();
public:
	FeatureMatching();
	FeatureMatching(
		cv::FlannBasedMatcher& matcher, cv::Ptr<cv::ORB> detector, cv::Mat descriptorImg, std::vector<cv::KeyPoint> keypointsImg, std::vector<cv::Point2f> markerCorners,
		int minMatchCount, float recognitionRatio, float correctionRatio,
		int maxCorners, float qualityLevel, int minDistance, int blockSize, float tracksRespawnRatio, float inlierThershold,
		cv::Mat markerMask
	);
	~FeatureMatching();
	bool track(bool& markerFound, std::vector<cv::Point2f> &matchedCorners, std::vector<cv::Point2f>& trackedCorners, std::vector<cv::Point2f>& tracks, int& tracksCount, cv::Mat& frameGray, cv::Mat maskBounding);
	cv::Mat & getHomography();
};

