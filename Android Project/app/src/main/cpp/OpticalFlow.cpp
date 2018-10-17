#include "OpticalFlow.h"

using namespace std;
using namespace cv;

OpticalFlow::OpticalFlow() {

}

OpticalFlow::OpticalFlow(Size winSize,int maxLevel, float tracksRespawnRatio, TermCriteria termCriteria, float backThreshold,
	int maxCorners, float qualityLevel, int minDistance, int blockSize, Mat markerMask)
{
	this->winSize = winSize;
	this->maxLevel = maxLevel;
	this->tracksRespawnRatio = tracksRespawnRatio;
	this->termCriteria = termCriteria;
	this->backThreshold = backThreshold;

	this->maxCorners = maxCorners;
	this->qualityLevel = qualityLevel;
	this->minDistance = minDistance;
	this->blockSize = blockSize;

	this->markerMask = markerMask;
}


OpticalFlow::~OpticalFlow()
{
}

void OpticalFlow::reset() {
	tracks.clear();
	tracksNew.clear();
	tracksRev.clear();
}

bool OpticalFlow::track(std::vector<cv::Point2f> & trackedCorners, Mat& framePrevGray, Mat& frameGray, int& tracksCount) {
	
	if (!tracks.empty()) {
		vector<uchar> status;
		if (prevPyr.empty())
			cv::buildOpticalFlowPyramid(framePrevGray, prevPyr, winSize, maxLevel, true);

		cv::buildOpticalFlowPyramid(frameGray, nextPyr, winSize, maxLevel, true);

		calcOpticalFlowPyrLK(prevPyr, nextPyr, tracks, tracksNew, trackStatus, err, winSize, maxLevel, termCriteria);
		swap(nextPyr, prevPyr);

		//Optical flow part
//		calcOpticalFlowPyrLK(framePrevGray, frameGray, tracks, tracksNew, trackStatus, err, winSize, maxLevel, termCriteria);
		//calcOpticalFlowPyrLK(frameGray, framePrevGray, tracksNew, tracksRev, trackStatus, err, winSize, maxLevel, termCriteria);

		///Tracks removal based on backTrack bayes filter
		/*vector<Point2f> shiftVectors;
		for (size_t i = 0; i < tracks.size(); i++) {
			shiftVectors.push_back(tracks[i] - tracksRev[i]);
		}*/

		////Utils::bayes(shiftVectors, status);
		//for (size_t i = 0; i < tracks.size(); i++) {
		//	if (Utils::euclideanDistance(shiftVectors[i]) < 0.2) {
		//		status.push_back(1);
		//	}
		//	else {
		//		status.push_back(1);
		//	}
		//}

		/*int i, k;
		vector<Point2f> outliers;
		for (k = 0, i = 0; i < tracks.size(); i++) {
			if (status[i] == 1) {
				tracks[k] = tracks[i];
				tracksNew[k] = tracksNew[i];
				k++;
			}
			else {
				outliers.push_back(shiftVectors[i]);
			}
		}
		tracks.resize(k);
		tracksNew.resize(k);*/
		vector<Point2f> shiftVectors;
		for (size_t i = 0; i < tracks.size(); i++) {
			shiftVectors.push_back(tracks[i] - tracksNew[i]);
		}

		Utils::bayes2D(shiftVectors, status);

		int i, k;
		vector<Point2f> outliers;
		for (k = 0, i = 0; i < tracks.size(); i++) {
			if (status[i] == 1) {
				tracks[k] = tracks[i];
				tracksNew[k] = tracksNew[i];
				k++;
			}
			/*else {
				outliers.push_back(shiftVectors[i]);
			}*/
		}
		tracks.resize(k);
		tracksNew.resize(k);

		/*Mat plot(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
		Utils::plotVectors(shiftVectors, plot, Scalar(0, 255, 0));
		Utils::plotVectors(outliers, plot, Scalar(0, 0, 255));
		imshow("plot", plot);*/

		if (tracks.size() < 8) {
			cerr << " < 8" << endl;
			return false;
		}

		Mat inlierMask;
		homography = findHomography(tracks, tracksNew, RANSAC, 10.0, inlierMask);

		///tracks removal based on RANSAC
		/*k = 0;
		for (unsigned i = 0; i < tracks.size(); i++) {
			if (inlierMask.at<uchar>(i)) {
				tracksNew[k] = tracksNew[i];
				k++;
			}
		}
		tracksNew.resize(k);*/

		perspectiveTransform(trackedCorners, trackedCorners, homography);

		if (!isContourConvex(trackedCorners)) {
			trackedCorners.clear();
			cerr << "convex" << endl;
		}

		tracks = tracksNew;
		return true;
	}
	else {
		trackedCorners.clear();
		return true;
	}
	
}