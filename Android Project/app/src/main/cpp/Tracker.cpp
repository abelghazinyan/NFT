#include "Tracker.h"
using namespace std;
using namespace cv;

Tracker::Tracker()
{
}

Tracker::Tracker(Mat &img):
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

    detector = ORB::create(FEATURE_COUNT / 3);

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

Tracker * Tracker::instance = 0;

Tracker *Tracker::getInstance(cv::Mat &marker) {
    if (instance == 0)
    {
        instance = new Tracker(marker);
    }

    return instance;
}

void Tracker::reset() {
    matchedCorners.clear();
    trackedCorners.clear();
    calculatedCorners.clear();
    predictedCorners.clear();
    opticalFlow.reset();
    markerFound = false;
}

void Tracker::matchingThread() {
    while (!stopped) {
        std::unique_lock<std::mutex> locker1(mtx1);
        frameGray.copyTo(tmpCur);
            featureMatching.track(markerFound, matchedCorners, trackedCorners,
                                  opticalFlow.tracks,
                                  tracksCount, tmpCur, maskBounding);
        condVar2.notify_one();
        condVar1.wait(locker1);
    }
}

void Tracker::start() {
    matchingT = thread(&Tracker::matchingThread, this);
    stopped = false;
}

void Tracker::stop() {
    stopped = true;
}

void Tracker::join() {
    matchingT.join();
}

bool Tracker::track(Mat &frame, vector<Point2f> &corners, bool debug)
{
    framesPassed++;
    matchedCorners.clear();
    calculatedCorners.clear();
    frameGray = frame;

//    cvtColor(frame, frameGray, COLOR_RGB2GRAY);

    if (framesPassed == MATCHING_PER_FRAME)
        condVar1.notify_one();
    ///OpticalFlow
        if (!trackedCorners.empty()) {
            if (!opticalFlow.track(trackedCorners, framePrevGray, frameGray, tracksCount)) {
                reset();
            }
        }
    ///Matching

    if (framesPassed == MATCHING_PER_FRAME) {
        std::unique_lock<std::mutex> locker2(mtx2);
        framesPassed = 0;
        condVar2.wait(locker2);
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
        if ((opticalFlow.tracks.size() < tracksCount*TRACKS_RESPAWN_RATIO) || (opticalFlow.tracks.size() < MIN_TRACK_COUNT)) {
            Mat maskPersp;
            Utils::maskPerspective(frameGray, markerMask, maskPersp, featureMatching.getHomography(), calculatedCorners, true);
            goodFeaturesToTrack(frameGray, opticalFlow.tracks, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, maskPersp, BLOCK_SIZE);
            tracksCount = opticalFlow.tracks.size();
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


    if (!calculatedCorners.empty()) {
        Utils::maskBoundingBox(frame, maskBounding, calculatedCorners);
        trackedCorners = calculatedCorners;
    }

    if (!predictedCorners.empty()) {
        if (debug)
            Utils::drawBoundingBox(frame, predictedCorners, Scalar(255, 0, 255, 100));
    }

    if (!calculatedCorners.empty()) {
        Utils::drawBoundingBox(frame, calculatedCorners, Scalar(0, 255, 0, 50));
    }

    frameGray.copyTo(framePrevGray);
    corners = calculatedCorners;

    return markerFound;
}

Mat & Tracker::getMarker() {
    return marker;
}

Tracker::~Tracker()
{
}
