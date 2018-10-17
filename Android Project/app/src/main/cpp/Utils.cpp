#include "Utils.h"

using namespace std;
using namespace cv;

Utils::Utils()
{
}


Utils::~Utils()
{
}

float Utils::getInlierRatio(vector<KeyPoint> matches, Mat inlierMask)
{
	vector<KeyPoint> inliers1, inliers2;
	vector<DMatch> inlier_matches;
	float ratio;

	for (unsigned i = 0; i < matches.size(); i++) {
		if (inlierMask.at<uchar>(i)) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matches[i]);
			inliers2.push_back(matches[i]);
			inlier_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	ratio = (int)inliers1.size() * 1.0 / (int)matches.size();
	return ratio;
}

int Utils::orientation(Point2f a, Point2f b, Point2f c)
{
	float k;
	k = (b.y - a.y)*(c.x - b.x) - (b.x - a.x) * (c.y - b.y);

	if (k > 0)
		return 1;
	else if (k < 0)
		return -1;
	return 0;
}

float Utils::trigSurface(Point2f a, Point2f b, Point2f c)
{
	float area_triangle;
	float A, B, C, P;

	A = (float)sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
	B = (float)sqrt((b.x - c.x)*(b.x - c.x) + (b.y - c.y)*(b.y - c.y));
	C = (float)sqrt((c.x - a.x)*(c.x - a.x) + (c.y - a.y)*(c.y - a.y));

	P = (A + B + C) / 2;

	area_triangle = (float)sqrt((P*(P - A)*(P - B)*(P - C)));

	return area_triangle;
}

void Utils::maskBoundingBox(Mat &frame, Mat &mask, std::vector<Point2f> &sceneCorners)
{
	Mat black = Mat::zeros(frame.size().height, frame.size().width, CV_8U);

	Rect rect = boundingRect(sceneCorners);

	rectangle(black, rect.tl(), rect.br(), Scalar(255, 255, 255), FILLED, 8, 0);
	black.copyTo(mask);
}

///-- Removes white background of marker with Canny finding biggest area contour
void Utils::removeBackground(Mat & input, Mat & outputArray)
{
	const float ratio = 0.9; ///-- Determining if marker has background

	Mat1b edges, thresh;
	Canny(input, edges, 100, 255);
	vector<vector<Point>> contours;
	findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	vector<int> indices(contours.size());
	iota(indices.begin(), indices.end(), 0);

	sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
		return contourArea(contours[lhs]) > contourArea(contours[rhs]);
	});

	Mat blankEdge = Mat::zeros(edges.size(), CV_8U);
	drawContours(blankEdge, contours, indices[0], Scalar(255, 255, 255), FILLED);
	blankEdge.copyTo(outputArray);

	///-- Thresholding then calculating area of contour for determining if it's with white bg or not
	vector<vector<Point>> contoursThresh;
	threshold(input, thresh, 250, 255, THRESH_BINARY_INV);
	findContours(thresh.clone(), contoursThresh, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);

	vector<int> indiciesThresh(contoursThresh.size());
	iota(indiciesThresh.begin(), indiciesThresh.end(), 0);

	sort(indiciesThresh.begin(), indiciesThresh.end(), [&contoursThresh](int lhs, int rhs) {
		return contourArea(contoursThresh[lhs]) > contourArea(contoursThresh[rhs]);
	});

	Mat blankThresh = Mat::zeros(edges.size(), CV_8U);
	drawContours(blankThresh, contoursThresh, indiciesThresh[0], Scalar(255, 255, 255), FILLED);

	float areaRatio = contourArea(contours[indices[0]]) / contourArea(contoursThresh[indiciesThresh[0]]);

	if (areaRatio > ratio) {
		blankEdge.copyTo(outputArray);
	}
	else {
		Mat white(input.size(), CV_8U);
		white.setTo(Scalar(255, 255, 255));
		white.copyTo(outputArray);
	}
}

void Utils::maskPerspective(Mat &frame, Mat markerMask, Mat &outputArray, Mat homography, std::vector<Point2f> &sceneCorners, bool useCorners)
{
	Mat mask = Mat::zeros(frame.size(), CV_8UC1);
	if (!useCorners) {
		warpPerspective(markerMask, mask, homography, frame.size());
	}
	else {
		drawPerspectiveProjection(markerMask, mask, sceneCorners, frame.size());
	}
	mask.copyTo(outputArray);
}

//TODO use only vector multiplications
bool Utils::checkIfConvexHull(vector<Point2f> points)
{
	float s1, s2, s3, s4;

	for (int i = 0; i < 4; i++)
	{
		s4 = trigSurface(points[(i + 1) % 4], points[(i + 2) % 4], points[(i + 3) % 4]);
		s1 = trigSurface(points[i], points[(i + 1) % 4], points[(i + 2) % 4]);
		s2 = trigSurface(points[i], points[(i + 1) % 4], points[(i + 3) % 4]);
		s3 = trigSurface(points[i], points[(i + 2) % 4], points[(i + 3) % 4]);
		if ((int)(s1 + s2 + s3) == (int)s4)
			return false;
	}

	if (orientation(points[0], points[1], points[2]) != orientation(points[2], points[3], points[0]))
		return false;
	return true;
}

void Utils::drawPoints(vector<Point2f> &corners, Mat &img, Scalar color)
{
	auto iterator = corners.begin();
	for (; iterator != corners.end(); iterator++)
	{
		circle(img, (*iterator), 2, color, -1, LINE_4);
	}
}

///High cost but accurate
void Utils::crossCheckMatching(FlannBasedMatcher& matcher, const Mat& descriptors1, const Mat& descriptors2,
                               vector<DMatch>& filteredMatches12, float ratio, int knn)
{
    double maxDist = 0, minDist = 100, dist;
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;

    matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
    matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
    for (size_t m = 0; m < matches12.size(); m++)
    {
        bool findCrossCheck = false;
        for (size_t fk = 0; fk < matches12[m].size(); fk++)
        {
            DMatch forward = matches12[m][fk];
            for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
            {
                if (matches12.size() == 2) {
                    if (&matches12[1][bk] != NULL)
                    {
                        dist = matches12[1][bk].distance;
                        if (dist < minDist) minDist = dist;
                        if (dist > maxDist) maxDist = dist;
                    }
                }
            }
            for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if (backward.trainIdx == forward.queryIdx)
                {
                    if (backward.distance <= minDist * ratio)
                    {
                        filteredMatches12.push_back(forward);
                    }
                    findCrossCheck = true;
                    break;
                }
            }
            if (findCrossCheck) break;
        }
    }
}


void Utils::drawBoundingBox(Mat &view, std::vector<Point2f> &scene_corners, Scalar color)
{
	///-- Draw lines between the corners (the mapped object in the scene - cam )
	line(view, scene_corners[0], scene_corners[1], color, 8, LINE_AA);
	line(view, scene_corners[1], scene_corners[2], color, 8, LINE_AA);
	line(view, scene_corners[2], scene_corners[3], color, 8, LINE_AA);
	line(view, scene_corners[3], scene_corners[0], color, 8, LINE_AA);
}

void Utils::drawPerspectiveProjection(Mat &img, Mat &out, vector<Point2f> &scene_corners, Size size)
{
	std::vector<Point2f> img_corners(4);
	img_corners[0] = Point2f(0, 0);
	img_corners[1] = Point2f(img.cols - 1, 0);
	img_corners[2] = Point2f(img.cols - 1, img.rows - 1);
	img_corners[3] = Point2f(0, img.rows - 1);
	Mat lambda(2, 4, CV_32FC1);
	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(img_corners, scene_corners);
	// Apply the Perspective Transform just found to the src image
	warpPerspective(img, out, lambda, size);
}

void Utils::plotVectors(vector<Point2f>  vectors, Mat & plot, Scalar color)
{
	for (size_t i = 0; i < vectors.size(); i++) {
		vectors[i] = vectors[i] * 5 + Point2f(plot.size().width / 2, plot.size().height / 2);
	}
	line(plot, Point2f(plot.size().width / 2, 0), Point2f(plot.size().width / 2, plot.size().height), Scalar(0, 0, 0), 1, LINE_AA);
	line(plot, Point2f(0, plot.size().height / 2), Point2f(plot.size().width, plot.size().height / 2), Scalar(0, 0, 0), 1, LINE_AA);
	Utils::drawPoints(vectors, plot, color);
}

///High cost but accurate
//void Utils::crossCheckMatching(FlannBasedMatcher& matcher, const Mat& descriptors1, const Mat& descriptors2,
//	vector<DMatch>& filteredMatches12, float ratio, int knn)
//{
//	double maxDist = 0, minDist = 100, dist;
//	filteredMatches12.clear();
//	vector<vector<DMatch> > matches12, matches21;
//	matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
//	matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
//	for (size_t m = 0; m < matches12.size(); m++)
//	{
//		bool findCrossCheck = false;
//		for (size_t fk = 0; fk < matches12[m].size(); fk++)
//		{
//			DMatch forward = matches12[m][fk];
//			for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
//			{
//				if (matches12.size() == 2) {
//					if (&matches12[1][bk] != NULL)
//					{
//						dist = matches12[1][bk].distance;
//						if (dist < minDist) minDist = dist;
//						if (dist > maxDist) maxDist = dist;
//					}
//				}
//			}
//			for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
//			{
//				DMatch backward = matches21[forward.trainIdx][bk];
//				if (backward.trainIdx == forward.queryIdx)
//				{
//					if (backward.distance <= minDist * ratio)
//					{
//						filteredMatches12.push_back(forward);
//					}
//					findCrossCheck = true;
//					break;
//				}
//			}
//			if (findCrossCheck) break;
//		}
//	}
//}

float Utils::euclideanDistance(cv::Point2f &p1, cv::Point2f p2)
{
	return (float) sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

void Utils::bayes(vector<Point2f> & points, vector<uchar> & status)
{
	Mat  myu, sigma;
	gaussian(points, myu, sigma);

	for (size_t i = 0; i < points.size(); i++) {
		if (predict(points[i], myu, sigma) < BAYES_THRESHOLD) {
			status.push_back(0);
		}
		else {
			status.push_back(1);
		}
	}
}

void Utils::gaussian(vector<Point2f> & pts, Mat & myu, Mat & sigma)
{
	Mat samples;
	Mat(pts).reshape(1, pts.size()).convertTo(samples, CV_32F);
	samples = samples.t();

	Mat myuTmp(2, 1, CV_32F);
	Mat sigmaTmp = Mat::zeros(2, 2, CV_32F);

	reduce(samples, myuTmp, 2, REDUCE_AVG);

	for (int i = 0; i < samples.cols; i++) {
		sigmaTmp += (samples.col(i) - myuTmp)*(samples.col(i) - myuTmp).t();
	}

	sigmaTmp = sigmaTmp / pts.size();

	myuTmp.copyTo(myu);
	sigmaTmp.copyTo(sigma);
}

float Utils::predict(Point2f x, Mat myu, Mat sigma)
{
	Mat y;
	Mat(x).reshape(1, 1).convertTo(y, CV_32F);
	y = y.t();
	Mat mat = (y - myu).t()*sigma.inv()*(y - myu) / (-2.0);

	float p = (float)mat.at<float>(0, 0);
	p = exp(p) / 2.0 / PI;
	float det = (float)sqrt(sigma.at<float>(0, 0) * sigma.at<float>(1, 1) - sigma.at<float>(1, 0) * sigma.at<float>(0, 1));
	p /= det;
	return p;
}

void Utils::bayes2D(vector<Point2f> & points, vector<uchar> & status)
{
	Point2f  myu, sigma;
	gaussian2D(points, myu, sigma);

	for (size_t i = 0; i < points.size(); i++) {
		if (predict2D(points[i], myu, sigma) < BAYES_THRESHOLD) {
			status.push_back(0);
		}
		else {
			status.push_back(1);
		}
	}
}

void Utils::gaussian2D(vector<Point2f> & pts, Point2f & myu, Point2f & sigma)
{
	Point2f mean(0, 0);
	Point2f sigma2(0, 0);
	Point2f var;
	for (auto &point : pts) {
		mean = mean + point;
	}
	mean = Point2f(mean.x / pts.size(), mean.y / pts.size());

	for (auto &point : pts) {
		var = mean - point;
		sigma2 = sigma2 + Point2f(var.x * var.x, var.y * var.y);
	}

	sigma2 = Point2f(sigma2.x / pts.size(), sigma2.y / pts.size());

	swap(mean, myu);
	swap(sigma2, sigma);
}

float Utils::predict2D(Point2f x, Point2f & myup, Point2f & sigmap)
{
	float p1, p2;

	float myu[2], sigma[2];

	myu[0] = myup.x;
	myu[1] = myup.y;

	sigma[0] = sigmap.x;
	sigma[1] = sigmap.y;

	p1 = 1 / sqrt(CV_PI * 2 * sigma[0]) * exp(-(x.x - myu[0])*(x.x - myu[0]) / 2 / sigma[0]);
	p2 = 1 / sqrt(CV_PI * 2 * sigma[1]) * exp(-(x.y - myu[1])*(x.y - myu[1]) / 2 / sigma[1]);

	return p1*p2;
}


void Utils::draw3DAxis(PNP &pnp, Mat &src, Mat &frame)
{
	Point2f vectorStart;
	Point2f vectorEnd[3];

	pnp.project(Point3f(130 + src.cols / 2, src.rows / 2, 0), vectorEnd[0]);
	pnp.project(Point3f(0 + src.cols / 2, -130 + src.rows / 2, 0), vectorEnd[1]);
	pnp.project(Point3f(0 + src.cols / 2, src.rows / 2, -130), vectorEnd[2]);
	pnp.project(Point3f(src.cols / 2, src.rows / 2, 0), vectorStart);

	cv::arrowedLine(frame, vectorStart, vectorEnd[0], cv::Scalar(0, 0, 255), 3, LINE_AA);
	cv::arrowedLine(frame, vectorStart, vectorEnd[1], cv::Scalar(0, 255, 0), 3, LINE_AA);
	cv::arrowedLine(frame, vectorStart, vectorEnd[2], cv::Scalar(255, 0, 0), 3, LINE_AA);
}

void Utils::drawCube(PNP &pnp, Mat &src, Mat &frame)
{
	Point2f vectorStart[4];
	Point2f vectorEnd[4];
	float depth = 100;
	pnp.project(Point3f(0, 0, -depth), vectorEnd[0]);
	pnp.project(Point3f(src.cols, 0, -depth), vectorEnd[1]);
	pnp.project(Point3f(src.cols, src.rows, -depth), vectorEnd[2]);
	pnp.project(Point3f(0, src.rows, -depth), vectorEnd[3]);
	pnp.project(Point3f(0, 0, 0), vectorStart[0]);
	pnp.project(Point3f(src.cols, 0, 0), vectorStart[1]);
	pnp.project(Point3f(src.cols, src.rows, 0), vectorStart[2]);
	pnp.project(Point3f(0, src.rows, 0), vectorStart[3]);
	Scalar color(0, 255, 0);
	cv::line(frame, vectorStart[0], vectorEnd[0], color, 9, LINE_AA);
	cv::line(frame, vectorStart[1], vectorEnd[1], color, 9, LINE_AA);
	cv::line(frame, vectorStart[2], vectorEnd[2], color, 9, LINE_AA);
	cv::line(frame, vectorStart[3], vectorEnd[3], color, 9, LINE_AA);

	cv::line(frame, vectorEnd[0], vectorEnd[1], color, 9, LINE_AA);
	cv::line(frame, vectorEnd[1], vectorEnd[2], color, 9, LINE_AA);
	cv::line(frame, vectorEnd[2], vectorEnd[3], color, 9, LINE_AA);
	cv::line(frame, vectorEnd[3], vectorEnd[0], color, 9, LINE_AA);

	cv::line(frame, vectorStart[0], vectorStart[1], color, 9, LINE_AA);
	cv::line(frame, vectorStart[1], vectorStart[2], color, 9, LINE_AA);
	cv::line(frame, vectorStart[2], vectorStart[3], color, 9, LINE_AA);
	cv::line(frame, vectorStart[3], vectorStart[0], color, 9, LINE_AA);
}