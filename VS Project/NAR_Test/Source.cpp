#include <iostream>
#include <vector>
#include <cmath>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "Utils.h"
#include "Tracker.h" 

using namespace std;
using namespace cv;

#define RESET_ANGLE 82

VideoCapture cap(0);

PNP pnp;

bool debug = false;
bool enable3D = true;

Mat marker, frame;

vector<Point2f> corners;   


int main()
{
	setUseOptimized(true);

	marker = imread("1.jpg", IMREAD_UNCHANGED);

	cerr << getBuildInformation() << endl;

	Tracker tracker(marker);
	marker = tracker.getMarker();
	pnp = PNP(marker, cap.get(CAP_PROP_FRAME_WIDTH));
	Size size = Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
	pnp.setCenter(size);

	double startTime, timePassed = 0;
	int frameCount = 0, fpsAvrg = 0;
	///Tracking loop
	for (;;) {

		if (frameCount == 10) {
			fpsAvrg = 10 / timePassed;
			timePassed = 0;
			frameCount = 0;	
		}

		startTime = (double)getTickCount();

		cap >> frame;

		tracker.track(frame, corners, debug);

		if (enable3D && !corners.empty()) {
			pnp.setCorners(corners);
			pnp.solve();
			//Utils::draw3DAxis(pnp, marker, frame);
			Utils::drawCube(pnp, marker, frame);
			Vec3f angles = pnp.getEuler();
			if (
				angles[0] > RESET_ANGLE || angles[0] < -RESET_ANGLE
				|| angles[1] > RESET_ANGLE || angles[1] < -RESET_ANGLE
				) {
				tracker.reset();
			}
		}

		//putText(frame, to_string(fpsAvrg), Point(2, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, LINE_4);

		imshow("Camera", frame);

		int key = waitKey(1);
		if (key == 27) break;
		if (key == 100) debug = !debug;
		if (key == 114) tracker.reset();
		if (key == 112) enable3D = !enable3D;
		//waitKey(-1);

		frameCount++;
		timePassed = timePassed + ((getTickCount() - startTime) / getTickFrequency());
	}
	return 0;
} 