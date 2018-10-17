#include <jni.h>
#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"
#include <string.h>
#include "Utils.h"
#include "Tracker.h"

#define RESET_ANGLE 82

using namespace std;
using namespace cv;

#define FRAME_HIEGHT 324

Ptr<ORB> detector;
PNP pnp;
Tracker* tracker;
bool debug = false;
bool enable3D = false;
bool sizeInited = false;
Size frameSize;
float scale, width;
Mat marker;
vector<Point2f> corners;

////
extern "C" JNIEXPORT void

JNICALL
Java_com_arloopa_abel_opencv_MainActivity_init(
        JNIEnv *env,
        jobject /* this */,
        jlong markerAdddr
) {
    setUseOptimized(true);
    marker = *(Mat*) markerAdddr;
    tracker = Tracker::getInstance(marker);
    marker = tracker->getMarker();
    pnp = PNP(marker, 1920);
    tracker->start();
}

extern "C" JNIEXPORT jstring

JNICALL
Java_com_arloopa_abel_opencv_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = getBuildInformation();
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void

JNICALL
Java_com_arloopa_abel_opencv_MainActivity_track(
        JNIEnv *env,
        jobject /* this */,
        jlong addrGray,
        jlong addrRgba
) {
    Mat &frame = *(Mat*) addrRgba;
    Mat &frameGray = *(Mat*) addrGray;

    if (!sizeInited) {
        width = frame.cols;
        pnp.setFocalLen(width);
        Size size = Size(frame.cols, frame.rows);
        pnp.setCenter(size);
        if (frame.rows > FRAME_HIEGHT) {
            frameSize = Size(frame.size().width * FRAME_HIEGHT / frame.size().height, FRAME_HIEGHT);
        }

        scale = width / frameSize.width;

        sizeInited = true;
    }

    Mat resized;

    resize(frameGray, resized, frameSize, INTER_LINEAR);

    tracker->track(resized, corners, debug);


    if (!corners.empty()) {
        for (auto &corner : corners) {
            corner.x = corner.x * scale;
            corner.y = corner.y * scale;
        }

        Utils::drawBoundingBox(frame, corners, Scalar(0, 0, 255));
    }

    if (enable3D && !corners.empty()) {
        pnp.setCorners(corners);
        pnp.solve();
//        Utils::draw3DAxis(pnp, marker, frame);
        Utils::drawCube(pnp, marker, frame);
        Vec3f angles = pnp.getEuler();
        if (
                angles[0] > RESET_ANGLE || angles[0] < -RESET_ANGLE
                || angles[1] > RESET_ANGLE || angles[1] < -RESET_ANGLE
                ) {
            tracker->reset();
        }
    }

}