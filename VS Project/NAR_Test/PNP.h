#pragma once

#include <iostream>
#include <algorithm>
#include <list>
#include <string>
#include <cmath>
#include <set>
#include "opencv2\core.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\opencv.hpp"

class PNP
{
private:
	int iterationsCount = 500;
	float reprojectionError = 2.0, confidence = 0.95;

	std::vector<cv::Point2f> imagePoints;
	std::vector<cv::Point3f> modelPoints;
	double focalLength;
	cv::Point2f center;
	cv::Mat cameraMatrix, camRotInv, distCoeffs;

	cv::Mat rotationVector, rotationMatrix; // Rotation in axis-angle form
	cv::Mat translationVector;
	bool isRotationMatrix(cv::Mat &R);
public:
	PNP();
	PNP(cv::Mat &, float);
	~PNP();
	void setCorners(std::vector<cv::Point2f>);
	void solve();
	void setFocalLen(float);
	float getFocalLen();
	void setCenter(cv::Size &);
	void solveRansac(std::vector<cv::Point3f> &,        // list with model 3D coordinates
		std::vector<cv::Point2f> &,        // list with scene 2D coordinates
		cv::Mat &);           // Ransac parameters
	void project(cv::Point3f, cv::Point2f &);
	void projectCorners(std::vector<cv::Point3f>, std::vector<cv::Point2f> &);
	cv::Mat & getRotMatrix();
	cv::Mat & getTransMatrix();
	void setTransMatrix(cv::Mat);
	void getSceneCorners(std::vector<cv::Point2f> &);
	cv::Vec3f getEuler();
};

