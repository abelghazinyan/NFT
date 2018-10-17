#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <cmath>
#include <numeric>

class Filter
{
private:
	std::vector<cv::Point2f> prevCorners, prevD1;  
public:
	Filter();
	~Filter();
	void init(std::vector<cv::Point2f> corners);
	void predict(std::vector<cv::Point2f> corners, std::vector<cv::Point2f> &predicted, double dt);
};

