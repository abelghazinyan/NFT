#pragma once

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "PNP.h"

#include <vector>
#include <cmath>
#include <numeric>

#define BAYES_THRESHOLD 0.00005f
#define PI 3.14159f


class Utils
{
private:
	static int orientation(cv::Point2f a, cv::Point2f b, cv::Point2f c);
	static float trigSurface(cv::Point2f a, cv::Point2f b, cv::Point2f c);
	static void gaussian(std::vector<cv::Point2f> &, cv::Mat &, cv::Mat &);
	static float predict(cv::Point2f, cv::Mat, cv::Mat);
	static void gaussian2D(std::vector<cv::Point2f> &, cv::Point2f &, cv::Point2f &);
	static float predict2D(cv::Point2f, cv::Point2f &, cv::Point2f &);

public:
	Utils();
	~Utils();
	static bool checkIfConvexHull(std::vector<cv::Point2f> points);
	static float getInlierRatio(std::vector<cv::KeyPoint> matches, cv::Mat inlierMask);
	static void maskBoundingBox(cv::Mat &frame, cv::Mat &mask, std::vector<cv::Point2f> &sceneCorners);
	static void maskPerspective(cv::Mat &frame, cv::Mat markerMask, cv::Mat &outputArray, cv::Mat homography, std::vector<cv::Point2f> &sceneCorners, bool = false);
	static void removeBackground(cv::Mat & input, cv::Mat & outputArray);
	static void drawPoints(std::vector<cv::Point2f> &corners, cv::Mat &img, cv::Scalar color);
	static void drawBoundingBox(cv::Mat &view, std::vector<cv::Point2f> &scene_corners, cv::Scalar color);
	static void drawPerspectiveProjection(cv::Mat &img, cv::Mat &out, std::vector<cv::Point2f> &scene_corners, cv::Size size);
	static void plotVectors(std::vector<cv::Point2f>  vectors, cv::Mat & plot, cv::Scalar color);
	static void crossCheckMatching(cv::FlannBasedMatcher &matcher, const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& filteredMatches12, float ratio, int knn = 2);
	static float euclideanDistance(cv::Point2f &p1, cv::Point2f p2 = cv::Point2f(0,0));
	static void bayes(std::vector<cv::Point2f> & points, std::vector<uchar> & status);
	static void bayes2D(std::vector<cv::Point2f> & points, std::vector<uchar> & status);
	static void draw3DAxis(PNP &pnp, cv::Mat &src, cv::Mat &frame);
	static void drawCube(PNP &pnp, cv::Mat &src, cv::Mat &frame);
};