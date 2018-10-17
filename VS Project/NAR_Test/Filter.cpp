#include "Filter.h"
#include "Utils.h"

using namespace std;
using namespace cv;

Filter::Filter()
{
}


Filter::~Filter()
{
}

void Filter::init(std::vector<cv::Point2f> corners) 
{
	this->prevCorners = corners;
	prevD1.clear();
}

void Filter::predict(vector<Point2f> corners, vector<Point2f> &predicted, double dt)
{
	double t = 1 / 30.0;
	vector<Point2f> d1;

	predicted.clear();

	for (size_t i = 0; i < corners.size(); i++) {
		d1.push_back((corners[i] - prevCorners[i]) / dt);
	}

	if (!prevD1.empty()) {
		for (size_t i = 0; i < corners.size(); i++) {
			Point2f pos;
			pos = ((d1[i] - prevD1[i]) / dt)*t*t / 2 + d1[i] * t + corners[i];
			predicted.push_back(pos);
		}
	} else{
		predicted.clear();
	}
	
	if (!predicted.empty()) {
		if (!isContourConvex(predicted)) {
			predicted.clear();
			prevCorners.clear();
		}
	}

	prevCorners = corners;
	prevD1 = d1;
}