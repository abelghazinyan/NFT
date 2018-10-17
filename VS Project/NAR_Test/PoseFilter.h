#pragma once

#include <iostream>
#include <algorithm>
#include <list>
#include <string>
#include <cmath>

#include "opencv2\core.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\opencv.hpp"

class PoseFilter
{
private:
	std::vector<cv::KalmanFilter> filter;
	std::vector<cv::Mat_<float>> measurement;
public:
	PoseFilter();
	~PoseFilter();
	void predict(std::vector<cv::Point2f>, double, std::vector<cv::Point2f> &);
	void correct(std::vector<cv::Point2f>, std::vector<cv::Point2f> &);
	void remove(std::vector<unsigned char>);
	void init(std::vector<cv::Point2f>);
};

