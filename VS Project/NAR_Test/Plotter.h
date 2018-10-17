#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <string>

#define COLOR_1 cv::Scalar(200, 0, 0)
#define COLOR_2 cv::Scalar(0, 200, 0)
#define COLOR_3 cv::Scalar(200, 0, 200)
#define COLOR_4 cv::Scalar(0, 255, 0)

class Plotter
{
private:
	std::string name;
	cv::Size size;
	int step;
	std::vector<cv::Point2f> prevPoints;

	int time;
	int cornerNumber;
	int max;
	cv::Mat blank;

	bool drawGrid = true;
	bool xOrY = false;
public:
	Plotter();
	Plotter(std::string name, cv::Size size, int step, bool xOrY, int cornerNumber, int max);
	~Plotter();
	void move();
	void draw(std::vector<cv::Point2f> &corners, cv::Scalar color, int num);
	void plot(std::vector<cv::Point2f> &corners1, std::vector<cv::Point2f> &corners2, std::vector<cv::Point2f> &corners3, std::vector<cv::Point2f> &corners4, cv::Mat & plot);
}; 

