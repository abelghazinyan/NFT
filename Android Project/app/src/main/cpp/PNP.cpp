#include "PNP.h"

using namespace std;
using namespace cv;

PNP::PNP()
{

}

PNP::PNP(Mat &img, float len)
{
	modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));
	modelPoints.push_back(cv::Point3d(img.size().width, 0.0f, 0.0f));
	modelPoints.push_back(cv::Point3d(img.size().width, img.size().height, 0.0f));
	modelPoints.push_back(cv::Point3d(0.0f, img.size().height, 0.0f));
	focalLength = len;
	distCoeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
}

PNP::~PNP()
{
}

void PNP::setCorners(vector<Point2f> corners)
{
	imagePoints = corners;
}

void PNP::solve()
{
	solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);
	Rodrigues(rotationVector, rotationMatrix);
	camRotInv = (cameraMatrix * rotationMatrix).inv();
}

void PNP::solveRansac(std::vector<cv::Point3f> &list_points3d, std::vector<cv::Point2f> &list_points2d, cv::Mat &inliers)
{
	bool useExtrinsicGuess = false;

	solvePnPRansac(list_points3d, list_points2d, cameraMatrix, distCoeffs, rotationVector, translationVector,
		useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
		inliers, SOLVEPNP_ITERATIVE);

	Rodrigues(rotationVector, rotationMatrix);
	camRotInv = (cameraMatrix * rotationMatrix).inv();
}

void PNP::project(Point3f point, Point2f &proj)
{
	vector<Point3f> points;
	vector<Point2f> projections;

	points.push_back(point);

	projectPoints(points, rotationVector, translationVector, cameraMatrix, distCoeffs, projections);
	proj = projections[0];
}

void PNP::projectCorners(vector<Point3f> points, vector<Point2f> &projections)
{
	projections.clear();
	Point2f projected;
	for (auto &pt3D : points) {
		project(pt3D, projected);
		projections.push_back(projected);
	}
}

void PNP::getSceneCorners(vector<Point2f> & corners)
{
	PNP::projectCorners(modelPoints, corners);
}

cv::Mat & PNP::getRotMatrix()
{
	return rotationVector;
}

cv::Mat & PNP::getTransMatrix()
{
	return translationVector;
}

void PNP::setTransMatrix(Mat mat)
{
	translationVector = mat;
}

void PNP::setFocalLen(float len)
{
	focalLength = len;
}

float  PNP::getFocalLen()
{
	return focalLength;
}

void PNP::setCenter(Size & size)
{
    center = cv::Point2f(size.width / 2, size.height / 2);
    cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength, 0, center.x, 0, focalLength, center.y, 0, 0, 1);
}

// Checks if a matrix is a valid rotation matrix.
bool PNP::isRotationMatrix(Mat &R)
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f PNP::getEuler()
{
	assert(isRotationMatrix(rotationMatrix));

	float sy = sqrt(rotationMatrix.at<double>(0, 0) * rotationMatrix.at<double>(0, 0) + rotationMatrix.at<double>(1, 0) * rotationMatrix.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(rotationMatrix.at<double>(2, 1), rotationMatrix.at<double>(2, 2));
		y = atan2(-rotationMatrix.at<double>(2, 0), sy);
		z = atan2(rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 0));
	}
	else
	{
		x = atan2(-rotationMatrix.at<double>(1, 2), rotationMatrix.at<double>(1, 1));
		y = atan2(-rotationMatrix.at<double>(2, 0), sy);
		z = 0;
	}
	return Vec3f(x * 180/CV_PI, y * 180 / CV_PI, z * 180 / CV_PI);
}
