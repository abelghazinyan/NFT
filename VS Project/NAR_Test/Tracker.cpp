#include "Tracker.h"

using namespace std;
using namespace cv;

Tracker::Tracker()
{
}

Tracker::Tracker(Mat &img):
	plot1("1st_X", Size(240, 500), 10, false, 0, 640),
	plot2("1st_Y", Size(240, 500), 10, true, 0, 640),
	plot3("2nd_X", Size(240, 500), 10, false, 1, 640),
	plot4("2nd_Y", Size(240, 500), 10, true, 1, 640),
	plot5("3th_X", Size(240, 500), 10, false, 2, 640),
	plot6("3th_Y", Size(240, 500), 10, true, 2, 640),
	plot7("4th_X", Size(240, 500), 10, false, 3, 640),
	plot8("4th_Y", Size(240, 500), 10, true, 3, 640),
	termCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03)
{
	img.copyTo(marker);
	/// Convert the image to grayscale
	cvtColor(marker, marker, COLOR_RGB2GRAY);
	/// Scale image to 512xX
	if (marker.cols > 512) {
		resize(marker, marker, Size(512, marker.size().height * 512 / marker.size().width));
	}

	markerCorners.push_back(Point(0, 0));
	markerCorners.push_back(Point(marker.cols, 0));
	markerCorners.push_back(Point(marker.cols, marker.rows));
	markerCorners.push_back(Point(0, marker.rows));
	///Get Mask without white background
	Utils::removeBackground(marker, markerMask);

	detector = ORB::create(100);
	detector->detectAndCompute(marker, noArray(), keypointsImg, descriptorImg);
	descriptorImg.convertTo(descriptorImg, CV_8UC1);

	matcher = FlannBasedMatcher(makePtr<cv::flann::LshIndexParams>(6, 12, 1));

	opticalFlow = OpticalFlow(
		WIN_SIZE, MAX_LEVEL, TRACKS_RESPAWN_RATIO, termCriteria, BACK_THRESHOLD,
		MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, BLOCK_SIZE, markerMask
	);

	featureMatching = FeatureMatching(
		matcher, detector, descriptorImg, keypointsImg, markerCorners,
		MIN_MATCH_COUNT, RECOGNITION_RATIO, CORRECTION_RATIO,
		MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, BLOCK_SIZE, TRACKS_RESPAWN_RATIO, INLIER_THRESHOLD,
		markerMask
	);
}

void Tracker::reset() {
	matchedCorners.clear();
	trackedCorners.clear();
	calculatedCorners.clear();
	predictedCorners.clear();
	opticalFlow.reset();
	markerFound = false;
}

bool Tracker::track(Mat& frame, vector<Point2f> &corners, bool debug)
{
	cvtColor(frame, frameGray, COLOR_RGB2GRAY);
	bool tryMatching = true;

	matchedCorners.clear();
	calculatedCorners.clear();

	///OpticalFlow
	if (!trackedCorners.empty()) {
		if (!opticalFlow.track(trackedCorners, framePrevGray, frameGray, tracksCount)) {
			reset();
		}
	}

	///Matching
	bool wasFound = markerFound;
	featureMatching.track(marker, markerFound, matchedCorners, trackedCorners, opticalFlow.tracks, tracksCount, frameGray, maskBounding);
	if (!wasFound && markerFound) {
		//filter.init(matchedCorners);
		//poseFilter.init(matchedCorners);
	}

	float wMatch = 0, wTrack = 0, wPred = 0;
	bool smooth = false;

	if (!matchedCorners.empty() && !trackedCorners.empty() && !predictedCorners.empty()) {
		wMatch = 0.5;
		wTrack = 0.5;
		wPred = 0;
		smooth = true;
	}
	else if (!matchedCorners.empty() && !trackedCorners.empty()) {
		wMatch = 0.5;
		wTrack = 0.5;
		smooth = true;
	}
	else if (!matchedCorners.empty() && !predictedCorners.empty()) {
		wMatch = 1;
		wPred = 0;
		smooth = true;
	}
	else if (!trackedCorners.empty() && !predictedCorners.empty()) {
		wTrack = 1;
		wPred = 0;
		smooth = true;
	}
	else if (!trackedCorners.empty()) {
		wTrack = 1;
		smooth = true;
	}
	else if (!matchedCorners.empty()) {
		wMatch = 1;
		smooth = true;
	}
	else {
		reset();
	}

	///Smoothing
	if (smooth) {
		for (size_t i = 0; i < 4; i++) {
			int n = 0;
			Point2f corner(0, 0);
			if (!matchedCorners.empty()) {
				corner.x += wMatch / matchedCorners[i].x;
				corner.y += wMatch / matchedCorners[i].y;
				n++;
			}
			if (!trackedCorners.empty()) {
				corner.x += wTrack / trackedCorners[i].x;
				corner.y += wTrack / trackedCorners[i].y;
				n++;
			}
			if (!predictedCorners.empty()) {
				corner.x += wPred / predictedCorners[i].x;
				corner.y += wPred / predictedCorners[i].y;
				n++;
			}

			corner.x = (wMatch + wTrack + wPred) / corner.x;
			corner.y = (wMatch + wTrack + wPred) / corner.y;
			calculatedCorners.push_back(corner);
		}
	}

	if (!calculatedCorners.empty()) {
		if (opticalFlow.tracks.size() < tracksCount*TRACKS_RESPAWN_RATIO) {
			Mat maskPersp;
			Utils::maskPerspective(frameGray, markerMask, maskPersp, featureMatching.getHomography(), calculatedCorners, true);
			goodFeaturesToTrack(frameGray, opticalFlow.tracks, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, maskPersp, BLOCK_SIZE);
			tracksCount = opticalFlow.tracks.size();
			cerr << "new corners" << endl;
		}

		///Remove outofcontour tracks TODO!  
		/*int k = 0;
		for (size_t i = 0; i < opticalFlow.tracks.size(); i++) {
		if (pointPolygonTest(calculatedCorners, opticalFlow.tracks[i], false) == 1) {
		opticalFlow.tracks[k] = opticalFlow.tracks[i];
		k++;
		}
		}
		opticalFlow.tracks.resize(k);*/
	}

	//Draw
	if (!opticalFlow.tracks.empty()) {
		if (debug)
			Utils::drawPoints(opticalFlow.tracks, frame, Scalar(0, 255, 255));
	}

	if (!trackedCorners.empty()) {
		if (debug)
			Utils::drawBoundingBox(frame, trackedCorners, Scalar(0, 0, 255));
	}

	if (!matchedCorners.empty()) {
		if (debug)
			Utils::drawBoundingBox(frame, matchedCorners, Scalar(255, 0, 0));
	}

	if (!calculatedCorners.empty()) {
		Utils::maskBoundingBox(frame, maskBounding, calculatedCorners);
		trackedCorners = calculatedCorners;
		//filter.predict(calculatedCorners, predictedCorners, ((double)getTickCount() - dt) / getTickFrequency());
		//poseFilter.correct(calculatedCorners, predictedCorners);
		//poseFilter.predict(calculatedCorners, ((double)getTickCount() - dt) / getTickFrequency(), predictedCorners);
	}

	if (!predictedCorners.empty()) {
		if (debug)
			Utils::drawBoundingBox(frame, predictedCorners, Scalar(255, 0, 255, 100));
	}

	if (!calculatedCorners.empty()) {
		if (debug)
			Utils::drawBoundingBox(frame, calculatedCorners, Scalar(0, 255, 0, 50));
	}

	frameGray.copyTo(framePrevGray);

	if (debug) {
		plot1.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p1);
		plot2.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p2);
		plot3.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p3);
		plot4.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p4);
		plot5.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p5);
		plot6.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p6);
		plot7.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p7);
		plot8.plot(matchedCorners, trackedCorners, predictedCorners, calculatedCorners, p8);

		hconcat(p1, p2, p2);
		hconcat(p3, p4, p4);
		hconcat(p5, p6, p6);
		hconcat(p7, p8, p8);

		vconcat(p2, p4, p4);
		vconcat(p4, p6, p6);
		vconcat(p6, p8, plot);

		if (!plot.empty())
			imshow("Plot", plot);
	}

	corners = calculatedCorners;

	return markerFound;
}

Mat & Tracker::getMarker() {
	return marker;
}

Tracker::~Tracker()
{
}
