#include "Plotter.h"

using namespace std;
using namespace cv;

Plotter::Plotter()
{
}

Plotter::Plotter(string name, cv::Size size, int step, bool xOrY, int cornerNumber, int max)
{
	this->name = name;
	this->size = size;
	this->step = step;

	Mat blank(size.width, size.height, 16);
	blank.setTo(Scalar(0, 0, 0));

	this->blank = blank;

	for (int i = 0; i < 4; i++) {
		prevPoints.push_back(Point2f(0,0));
	}

	this->xOrY = xOrY;
	this->cornerNumber = cornerNumber;
	this->max = max;
}

Plotter::~Plotter()
{
}

void Plotter::draw(std::vector<cv::Point2f> &corners, cv::Scalar color, int num)
{
	float data = xOrY ? corners[cornerNumber].y : corners[cornerNumber].x;

	data = -data * blank.rows / max + blank.rows;

	if (prevPoints[num] == Point2f(0, 0)) {
		circle(blank, Point2f(time, data), 1, color, -1, LINE_AA);
	}
	else {
		line(blank, prevPoints[num], Point2f(time, data), color, 1, LINE_AA);
		circle(blank, Point2f(time, data), 2, color, -1, LINE_AA);
	}
	prevPoints[num] = Point2f(time, data);
}

void Plotter::plot(vector<Point2f> &corners1, vector<Point2f> &corners2, vector<Point2f> &corners3, vector<Point2f> &corners4, Mat &plot) {

	if (drawGrid) {
		for (size_t i = 0; i < blank.cols; i += 10) {
			line(blank, Point2f(i, 0), Point2f(i, blank.rows), Scalar(20, 20, 20), 1, LINE_AA);
		}
		for (size_t i = 0; i < blank.rows; i += 10) {
			line(blank, Point2f(0, i), Point2f(blank.cols, i), Scalar(20, 20, 20), 1, LINE_AA);
		}
		drawGrid = !drawGrid;
	}

	if (time >= blank.cols) {
		blank.setTo(Scalar(0, 0, 0));
		time = 0;

		prevPoints.clear();
		for (int i = 0; i < 4; i++) {
			prevPoints.push_back(Point2f(0, 0));
		}

		for (size_t i = 0; i < blank.cols; i += 10) {
			line(blank, Point2f(i, 0), Point2f(i, blank.rows), Scalar(20, 20, 20), 1, LINE_AA);
		}

		for (size_t i = 0; i < blank.rows; i += 10) {
			line(blank, Point2f(0, i), Point2f(blank.cols, i), Scalar(20, 20, 20), 1, LINE_AA);
		}

		drawGrid = !drawGrid;
	}

	if (!corners1.empty())
		draw(corners1, COLOR_1, 0);
	if (!corners2.empty())
		draw(corners2, COLOR_2, 1);
	if (!corners3.empty())
		draw(corners3, COLOR_3, 2);
	if (!corners4.empty())
		draw(corners4, COLOR_4, 3);
	
	move();

	putText(blank, name, Point(0, 15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(180, 180, 180));
	blank.copyTo(plot);
	//imshow(name, blank);
}

void Plotter::move() {
	time += step;
}