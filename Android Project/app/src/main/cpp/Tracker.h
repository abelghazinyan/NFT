#pragma once

#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/flann.hpp"

#include "Utils.h"
#include "OpticalFlow.h"
#include "FeatureMatching.h"
#include "PNP.h"
#include "Plotter.h"


///Configs
///GoodFeautures
#define MIN_MATCH_COUNT 8
#define MAX_CORNERS 50
#define QUALITY_LEVEL 0.01f
#define MIN_DISTANCE 8
#define BLOCK_SIZE 19
///Lk params
#define WIN_SIZE Size(20,20)
#define MAX_LEVEL 2
#define TRACKS_RESPAWN_RATIO 0.55f
#define BACK_THRESHOLD 1.0f
///Matching Params
#define MATCHING_PER_FRAME 4 /// N - 1 frames beetween matches, 1 means every frame!
#define FEATURE_COUNT 300
#define RECOGNITION_RATIO 0.5f
#define CORRECTION_RATIO 0.75f
#define MIN_TRACK_COUNT 10
///
#define INLIER_THRESHOLD 0.7f

class Tracker
{
private:
    cv::TermCriteria termCriteria;

    cv::Ptr<cv::ORB> detector;
    cv::FlannBasedMatcher matcher;

    std::vector<cv::KeyPoint> keypointsImg;

    cv::Mat descriptorImg;
    cv::Mat maskBounding, frameGray, framePrevGray;

    std::vector<cv::Point2f> markerCorners;
    std::vector<cv::Point2f>  matchedCorners, trackedCorners, calculatedCorners, predictedCorners;

    cv::Mat marker, markerMask;

    bool markerFound = false;
    int tracksCount;

    OpticalFlow opticalFlow;
    FeatureMatching featureMatching;
    PNP pnp;

    std::thread matchingT, optFlowT;
    void matchingThread();
    void optFlowThread();
    cv::Mat tmpFrameGray, tmpMask, tmpPrev, tmpCur;
    static Tracker * instance;
    bool stopped = false;
    bool ready = false, processed = false;
    std::mutex mtx1, mtx2, mtx3;
    std::condition_variable condVar1, condVar2, condVar3;
    int framesPassed = 0;
    int framesSinceOptFlow = 0;
    Tracker();
    Tracker(cv::Mat &marker);
public:
    static Tracker* getInstance(cv::Mat &marker);
    bool track(cv::Mat &frame, std::vector<cv::Point2f> &corners, bool debug);
    ~Tracker();
    void reset();
    void start();
    void stop();
    void join();
    cv::Mat & getMarker();
};

