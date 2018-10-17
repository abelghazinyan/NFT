#include "FeatureMatching.h"

using namespace std;
using namespace cv;

FeatureMatching::FeatureMatching()
{
}

FeatureMatching::FeatureMatching(
	FlannBasedMatcher& matcher, Ptr<ORB> detector, Mat descriptorImg, vector<KeyPoint> keypointsImg, vector<Point2f> markerCorners,
	int minMatchCount, float recognitionRatio, float correctionRatio,
	int maxCorners, float qualityLevel, int minDistance, int blockSize, float tracksRespawnRatio, float inlierThershold,
	Mat markerMask
)
{
	this->matcher = matcher;
	this->detector = detector;
	this->descriptorImg = descriptorImg;
	this->keypointsImg = keypointsImg;
	this->markerCorners = markerCorners;
	this->minMatchCount = minMatchCount;
	this->recognitionRatio = recognitionRatio;
	this->correctionRatio = correctionRatio;
	this->maxCorners = maxCorners;
	this->qualityLevel = qualityLevel;
	this->minDistance = minDistance;
	this->blockSize = blockSize;
	this->tracksRespawnRatio = tracksRespawnRatio;
	this->inlierThershold = inlierThershold;
	this->markerMask  = markerMask;
}

FeatureMatching::~FeatureMatching()
{

}

void FeatureMatching::reset() {

}

bool FeatureMatching::track(Mat &marker, bool& markerFound, vector<Point2f> &matchedCorners, vector<Point2f>& trackedCorners, vector<Point2f>& tracks, int& tracksCount, Mat& frameGray, Mat maskBounding)
{

	///Feature Matching
	keypointsDetected.clear();
	descriptorDetected.release();
	goodMatches.clear();
	matchedCorners.clear();
	matches.clear();
	//Mask tracking
	if (markerFound) {
		detector->setMaxFeatures(100);
		/*detector->detect(frameGray, keypointsDetected, maskBounding);
		detector->compute(frameGray, keypointsDetected, descriptorDetected);*/
		detector->detectAndCompute(frameGray, maskBounding, keypointsDetected, descriptorDetected, false);
	}
	else {
		detector->setMaxFeatures(500);
		/*detector->detect(frameGray, keypointsDetected);
		detector->compute(frameGray, keypointsDetected, descriptorDetected);*/
		detector->detectAndCompute(frameGray, noArray(), keypointsDetected, descriptorDetected);
	}
	// Convert to FLANN format
	descriptorDetected.convertTo(descriptorDetected, CV_8UC1);
	matchedKeypoints.clear();
	obj.clear();
	scene.clear();

	if (keypointsDetected.size() < minMatchCount) {
		return false;
	}

	if (!descriptorDetected.empty() && !descriptorImg.empty()) {

		if (keypointsDetected.size() >= minMatchCount) {
			if (markerFound) {
				matcher.knnMatch(descriptorImg, descriptorDetected, matches, 2);
				for (auto &match : matches) {
					if ((match.size() == 2)) {
						if ((match[0].distance < match[1].distance * correctionRatio))
							goodMatches.push_back(match[0]);
						else if ((match[1].distance < match[0].distance * correctionRatio)) {
							goodMatches.push_back(match[1]);
						}
					}
				}
				//Utils::crossCheckMatching(matcher, descriptorImg, descriptorDetected, goodMatches, correctionRatio);
			}
			else {
				Utils::crossCheckMatching(matcher, descriptorImg, descriptorDetected, goodMatches, recognitionRatio);
			}
		}


		if (goodMatches.size() < minMatchCount) {
			return false;
		}

		/*Mat matchesImg = Mat();
		drawMatches(marker, keypointsImg, frameGray, keypointsDetected,
			goodMatches, matchesImg, Scalar(0, 255, 255), Scalar(0, 255, 255),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("matches", matchesImg);*/


		for (size_t i = 0; i < goodMatches.size(); i++)
		{
			matchedKeypoints.push_back(keypointsImg[goodMatches[i].queryIdx]);
			obj.push_back(keypointsImg[goodMatches[i].queryIdx].pt);
			scene.push_back(keypointsDetected[goodMatches[i].trainIdx].pt);
		}
		
		Mat inlierMask;
		homography = findHomography(obj, scene, RANSAC, 3.0, inlierMask);

		int inliers = 0;
		for (int i = 0; i < inlierMask.rows; i++) {
			inliers += inlierMask.at<uchar>(i);
		}

		if (inliers < minMatchCount) {
			if (!markerFound) {
				reset();
				return false;
			}
			else {
				return false;
			}
		}

		/// Calculate Inlier Ratio
		float inlierRatio = Utils::getInlierRatio(matchedKeypoints, inlierMask);
		//cerr << inlierRatio << endl;
		if (inlierRatio < inlierThershold)
		{
			reset();
			return false;
		}

		if (!homography.size().empty())
		{
			if (!markerFound)
				trackedCorners.clear();

			perspectiveTransform(markerCorners, matchedCorners, homography);

			/// Check if image is not scewed	
			if (!isContourConvex(matchedCorners))
			{
				if (!markerFound) {
					//cerr << "matching not convex" << endl;
					matchedCorners.clear();
					trackedCorners.clear();
					markerFound = false;
					tracks.clear();
					return false;
				}
				else {
					matchedCorners.clear();
				}
			}

			/*if (found) {
			matchedCorners.clear();
			}*/

			if (!markerFound) {
				markerFound = true;
				Mat maskPersp;
				Utils::maskPerspective(frameGray, markerMask, maskPersp, homography, matchedCorners);
				goodFeaturesToTrack(frameGray, tracks, maxCorners, qualityLevel, minDistance, maskPersp, blockSize);
				tracksCount = tracks.size();
				trackedCorners = matchedCorners;
				cerr << "new corners matching" << endl;
			}
		}
	}

	return true;
}

Mat & FeatureMatching::getHomography() {
	return homography;
}