#include "PoseFilter.h"

using namespace std;
using namespace cv;

PoseFilter::PoseFilter()
{

}

void PoseFilter::init(vector<Point2f> points)
{
	filter.clear();
	measurement.clear();
	// init...
	for (size_t i = 0; i < points.size(); i++)
	{
		filter.push_back(KalmanFilter(4, 2, 1, CV_32F));
		filter[i].transitionMatrix =
			(Mat_<float>(4, 4) <<
				1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1
				);
		//setIdentity(filter[i].transitionMatrix);
		measurement.push_back(Mat::zeros(2, 1, CV_32F));
		filter[i].statePre.setTo(0);
		filter[i].controlMatrix.setTo(0);

		//initialzing filter 
		filter[i].statePre.at<float>(0) = points[i].x; //the first reading
		filter[i].statePre.at<float>(1) = points[i].y;
		filter[i].statePre.at<float>(2) = 0;
		filter[i].statePre.at<float>(3) = 0;

		setIdentity(filter[i].measurementMatrix);
		setIdentity(filter[i].processNoiseCov, Scalar::all(1e-3)); //updated at every step
		setIdentity(filter[i].measurementNoiseCov, Scalar::all(1e-5)); //assuming measurement error of  //not more than 2 pixels  

		setIdentity(filter[i].errorCovPost, Scalar::all(1e-5));

		/*filter.push_back(KalmanFilter(4, 2, 0, CV_32F));
		filter[i].transitionMatrix =
		(Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1
		);
		measurement.push_back(Mat_<float>(2, 1));
		measurement[i].setTo(Scalar(0));

		filter[i].statePre.at<float>(0) = points[i].x;
		filter[i].statePre.at<float>(1) = points[i].y;
		filter[i].statePre.at<float>(2) = 0;
		filter[i].statePre.at<float>(3) = 0;


		filter[i].processNoiseCov =
		(Mat_<float>(4, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1

		);
		setIdentity(filter[i].measurementMatrix);
		setIdentity(filter[i].processNoiseCov, Scalar::all(10));
		setIdentity(filter[i].measurementNoiseCov, Scalar::all(10));
		setIdentity(filter[i].errorCovPost, Scalar::all(10));*/
	}
}

void PoseFilter::predict(vector<Point2f> points, double dt, vector<Point2f> & screenPts)
{
	screenPts.clear();
	for (size_t i = 0; i < points.size(); i++)
	{
		//// First predict, to update the internal statePre variable
		//Mat prediction = filter[i].predict();
		//Point2f predictPt(prediction.at<float>(0), prediction.at<float>(1));
		//screenPts.push_back(predictPt);

		//Updating the transitionMatrix
		filter[i].transitionMatrix.at<float>(0, 2) = dt;
		filter[i].transitionMatrix.at<float>(1, 3) = dt;

		//Updating the Control matrix
		filter[i].controlMatrix.at<float>(0, 1) = (dt*dt) / 2;
		filter[i].controlMatrix.at<float>(1, 1) = (dt*dt) / 2;
		filter[i].controlMatrix.at<float>(2, 1) = dt;
		filter[i].controlMatrix.at<float>(3, 1) = dt;

		//Updating the processNoiseCovmatrix
		filter[i].processNoiseCov.at<float>(0, 0) = (dt*dt*dt*dt) / 4;
		filter[i].processNoiseCov.at<float>(0, 2) = (dt*dt*dt) / 2;
		filter[i].processNoiseCov.at<float>(1, 1) = (dt*dt*dt*dt) / 4;
		filter[i].processNoiseCov.at<float>(1, 3) = (dt*dt*dt) / 2;

		filter[i].processNoiseCov.at<float>(2, 0) = (dt*dt*dt) / 2;
		filter[i].processNoiseCov.at<float>(2, 2) = dt * dt;

		filter[i].processNoiseCov.at<float>(3, 1) = (dt*dt*dt) / 2;
		filter[i].processNoiseCov.at<float>(3, 3) = dt * dt;

		Mat prediction1 = filter[i].predict();
		Point2f predictPt1(prediction1.at<float>(0), prediction1.at<float>(1));
		screenPts.push_back(predictPt1);
	}
}

void PoseFilter::correct(vector<Point2f> points, vector<Point2f> & corrected)
{
	corrected.clear();
	for (size_t i = 0; i < points.size(); i++)
	{
		measurement[i](0) = points[i].x;
		measurement[i](1) = points[i].y;

		// The "correct" phase that is going to use the predicted value and our measurement
		Mat estimated = filter[i].correct(measurement[i]);
		Point2f statePt(estimated.at<float>(0), estimated.at<float>(1));
		corrected.push_back(statePt);
	}
}

void PoseFilter::remove(std::vector<unsigned char> status)
{
	size_t k = 0, i;
	for (i = 0; i < status.size(); i++) {
		if (status[i] == 1) {
			filter[k] = filter[i];
			measurement[k] = measurement[i];
			k++;
		}
	}
	filter.resize(k);
	measurement.resize(k);
}

PoseFilter::~PoseFilter()
{
}
