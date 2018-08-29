#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

//MYNTEYE
#include <opencv2/highgui/highgui.hpp>
#include <mynteye/global.h>
#include <mynteye/api.h>

using namespace cv;
using namespace std;

MYNTEYE_USE_NAMESPACE

#define IMG_NUM 13

static void calObjectPosition(Size boardSize, float squareSize, vector<Point3f>& corners);
double ch_cameraCalibration(String* imageName, int imageNum, Size boardSize, float squareSize, 
	                        Mat& cameraMatrix, Mat& distCoeffs, Mat& map1, Mat& map2);
void ch_undistort(InputArray src, OutputArray dst, InputArray map1, InputArray map2);

void ch_stereoCalibration();

int ch_findCorners(String* imageNames, int imageNum, Size boardSize, vector<vector<Point2f>>& imagePoints);

static vector<vector<Point3f>> ch_calObjectCorners(Size boardSize, float squqreSize, vector<vector<Point2f>> imageCorners);


int calibrationTest()
{
	Size boardSize = Size(9, 6);
	float squareSize = 50.0f;

	String img_names[IMG_NUM] = { "E:/WorkSpace_VisualStudio/Project2/data/left01.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left02.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left03.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left04.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left05.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left06.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left07.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left08.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left09.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left11.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left12.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left13.jpg" ,
								  "E:/WorkSpace_VisualStudio/Project2/data/left14.jpg" };


	Mat cameraMatrix;
	Mat distCoeffs;
	Mat map1, map2;
	ch_cameraCalibration(img_names, 13, boardSize, squareSize, cameraMatrix, distCoeffs, map1, map2);

	Mat v = imread(img_names[0]);
	Mat v1;
	ch_undistort(v, v1, map1, map2);
	imshow("distored", v1);
	waitKey(100);

	v = imread(img_names[2]);
	//Solve PNP
	vector<Point3f> objPoints;
	vector<Point2f> imgPoints;
	Mat rvec, tvec;
	calObjectPosition(boardSize, squareSize, objPoints);
	bool found = findChessboardCorners(v, boardSize, imgPoints, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	//Normal pnp
	int64 t0, t1;
	double dt;
	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_ITERATIVE);
	t1 = getTickCount();
	dt = ((double)(t1 - t0)) / getTickFrequency();
	cout << endl;
	cout << "PNP running time: " << dt << endl;
	cout << "rvec:\n" << rvec << endl;
	cout << "tvec:\n" << tvec << endl<<endl;
	//Effiction pnp
	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_EPNP);
	t1 = getTickCount();
	dt = ((double)(t1 - t0)) / getTickFrequency();
	cout << "EPNP running time: " << dt << endl;
	cout << "rvec:\n" << rvec << endl;
	cout << "tvec:\n" << tvec << endl<<endl;
	//Direct Least-Squares
	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_DLS);
	t1 = getTickCount();
	dt = ((double)(t1 - t0)) / getTickFrequency();
	cout << "DLS running time: " << dt << endl;
	cout << "rvec:\n" << rvec << endl;
	cout << "tvec:\n" << tvec << endl<<endl;
	//PNP RANSAC
	t0 = getTickCount();
	solvePnPRansac(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 8.0f, 0.99, noArray(), CV_EPNP);
	t1 = getTickCount();
	dt = ((double)(t1 - t0)) / getTickFrequency();
	cout << "PNP RANSAC running time: " << dt << endl;
	cout << "rvec:\n" << rvec << endl;
	cout << "tvec:\n" << tvec << endl << endl;
	
	Mat rm;
	Rodrigues(rvec, rm);
	cout << "rotation matrix:\n " << rm << endl;
	imshow("c", v);
	waitKey();

	return 0;

}

int ch_stereoTest()
{
	auto &&api = API::Create();

	if (!api)
	{
		cout << "Can not open MYNTEYE" << endl;
		return 1;
	}
	api->EnableStreamData(Stream::LEFT_RECTIFIED);
	api->EnableStreamData(Stream::RIGHT_RECTIFIED);
	//api->EnableStreamData(Stream::DEPTH);
	api->Start(Source::VIDEO_STREAMING);
	namedWindow("frame");
	//namedWindow("frame1");
	int pics_cnt = 0;
	while (true)
	{
		api->WaitForStreams();
		auto &&left_data = api->GetStreamData(Stream::LEFT_RECTIFIED);
		auto &&right_data = api->GetStreamData(Stream::RIGHT_RECTIFIED);
		//auto &&depth_data = api->GetStreamData(Stream::DEPTH);
		if (!left_data.frame.empty() && !right_data.frame.empty()) {
			Mat img;
			hconcat(left_data.frame, right_data.frame, img);
			imshow("frame", img);
		}
		//if (!depth_data.frame.empty())
		//{
		//	imshow("frame1", depth_data.frame);
		//}
		char key = static_cast<char>(cv::waitKey(1));
		if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
			break;
		}
		else if (key == 'p')
		{
			string sl, sr, nb;
			string dir = "E:/WorkSpace_VisualStudio/stereo_data/";
			nb = to_string(pics_cnt);
			sl = dir + "left" + nb + ".png";
			sr = dir + "right" + nb + ".png";
			imwrite(sl, left_data.frame);
			imwrite(sr, right_data.frame);
			cout << "Write picture " << pics_cnt << " done" << endl;
			pics_cnt++;
		}
	}
	api->Stop(Source::VIDEO_STREAMING);
	return 0;
}

void ch_getPictures()
{
	auto &&api = API::Create();
	if (!api)
	{
		cout << "Can not open camera" << endl;
		return;
	}

	api->EnableStreamData(Stream::LEFT);
	api->EnableStreamData(Stream::RIGHT);
	api->Start(Source::VIDEO_STREAMING);
	namedWindow("camera");

	int pics_cnt=0;

	while (true)
	{
		api->WaitForStreams();
		auto &&left_data = api->GetStreamData(Stream::LEFT);
		auto &&right_data = api->GetStreamData(Stream::RIGHT);

		if (!left_data.frame.empty() && !right_data.frame.empty())
		{
			Mat img;
			hconcat(left_data.frame, right_data.frame, img);
			imshow("camera", img);

			char key = static_cast<char>(waitKey(1));
			if (key == 27 || key == 'q' || key == 'Q')
			{
				break;
			}
			else if (key == 'p')
			{
				string sl, sr, nb;
				string dir = "E:/WorkSpace_VisualStudio/stereo_data/";
				nb = to_string(pics_cnt);
				sl = dir + "left" + nb + ".png";
				sr = dir + "right" + nb + ".png";
				imwrite(sl, left_data.frame);
				imwrite(sr, right_data.frame);
				cout << "Write picture " << pics_cnt << " done" << endl;
				pics_cnt++;
			}
		}

	}
}

int main()
{
//	ch_getPictures();
	ch_stereoCalibration();
//	ch_stereoTest();
}

#define STEREO_IMG_NUM 16
void ch_stereoCalibration()
{
	int num = STEREO_IMG_NUM;
	Size boardSize(9, 7);
	float squareSize = 23.3;
	string dir = "E:/WorkSpace_VisualStudio/stereo_data/";
	String left_name[STEREO_IMG_NUM], right_name[STEREO_IMG_NUM];
	int num_ok[17] = { 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 23, 29, 30, 31, 32 };
	for (int i = 0; i < STEREO_IMG_NUM; i++)
	{
		left_name[i] = dir + "left" + to_string(num_ok[i]) + ".png";
		right_name[i] = dir + "right" + to_string(num_ok[i]) + ".png";
	}

	Mat M1, M2, D1, D2, leftMap1, leftMap2, rightMap1, rightMap2;
	ch_cameraCalibration(left_name, num, boardSize, squareSize, M1, D1, leftMap1, leftMap2);
	ch_cameraCalibration(right_name, num, boardSize, squareSize, M2, D2, rightMap1, rightMap2);

	Mat v = imread(left_name[0]);
	vector<vector<Point2f>> leftCorners, rightCorners;
	vector<vector<Point3f>> objectCorners;
	// Stereo 
	Mat R, T, E, F;
	ch_findCorners(left_name, num, boardSize, leftCorners);
	ch_findCorners(right_name, num, boardSize, rightCorners);
	objectCorners = ch_calObjectCorners(boardSize, squareSize, leftCorners);
	stereoCalibrate(objectCorners, leftCorners, rightCorners, M1, D1, M2, D2,
		v.size(), R, T, E, F, CALIB_FIX_INTRINSIC);
	cout << "Left Matrix:\n" << M1 << endl;
	cout << "Left Coef:\n" << D1 << endl;
	cout << "Right Matrix:\n" << M2 << endl;
	cout << "Right Coef:\n" << D2 << endl;
	cout << "R:\n" << R << endl;
	cout << "T:\n" << T << endl;
	cout << "E:\n" << E << endl;
	cout << "F:\n" << F << endl;

	//Test
	Size img_size = v.size();
	Mat R1, R2, P1, P2, Q;
	Rect roi1, roi2;
	stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	Mat v1, v2;
	v1 = imread("E:/WorkSpace_VisualStudio/stereo_data/left1.png");
	v2 = imread("E:/WorkSpace_VisualStudio/stereo_data/right1.png");

	Mat v1n, v2n;
	remap(v1, v1n, map11, map12, INTER_LINEAR);
	remap(v2, v2n, map21, map22, INTER_LINEAR);
	cvtColor(v1n, v1, cv::COLOR_BGR2GRAY);
	cvtColor(v2n, v2, cv::COLOR_BGR2GRAY);
	//BM Setting
	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	int numberOfDisparities = 16;
	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(10);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	Mat disp, disp8;
	v1n.convertTo(v1n, CV_8U);
	v2n.convertTo(v2n, CV_8U);
	bm->compute(v1n, v2n, disp);
	//disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	imshow("left", v1n);
	imshow("right", v2n);
	//imshow("disparity", disp8);

	waitKey();

}

int ch_findCorners(String* imageNames, int imageNum, Size boardSize, vector<vector<Point2f>>& imagePoints)
{
	vector<Mat> views;
	int found_flag = 0;
	for (int i = 0; i < imageNum; i++)
	{
		vector<Point2f> pointBuf;
		Mat v;

		v = imread(imageNames[i]);
		views.push_back(v);

		bool found;
		found = findChessboardCorners(v, boardSize, pointBuf,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (found)
		{
			Mat viewGray;
			cvtColor(v, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			imagePoints.push_back(pointBuf);
			drawChessboardCorners(v, boardSize, Mat(pointBuf), found);
			found_flag++;
			cout << "found " << found_flag << endl;
			imshow("org", v);
			waitKey(20);
		}
	}
	return found_flag;
}

double ch_cameraCalibration(String* imageName, int imageNum, Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs, Mat& map1, Mat& map2)
{
	vector<vector<Point2f>> imagePoints;
	vector<Mat> views;

	int found_flag = 0;
	//Collecting Points
	for (int i = 0; i < imageNum; i++)
	{
		vector<Point2f> pointBuf;
		Mat v;

		v = imread(imageName[i]);
		views.push_back(v);
		//Finding corners
		bool found;
		found = findChessboardCorners(v, boardSize, pointBuf, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

		//Draw corners
		if (found)
		{
			Mat viewGray;
			cvtColor(v, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			imagePoints.push_back(pointBuf);
			drawChessboardCorners(v, boardSize, Mat(pointBuf), found);
			found_flag++;
			cout << "found " << found_flag << endl;
			imshow("org", v);
			waitKey(20);
		}
		
	}
	Size imageSize = views[0].size();

	vector<vector<Point3f>> objectPoints(1);
	calObjectPosition(boardSize, squareSize, objectPoints[0]);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Call opencv function
	double rms;
	vector<Mat> rvecs, tvecs;
	rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);

	cout << "rvecs: \n" << rvecs[0] << endl;
	cout << "tvecs: \n" << tvecs[0] << endl;
	cout << "Camera Matrix:\n" << cameraMatrix << endl;
	cout << "DistCoeffese: \n" << distCoeffs << endl;

	Mat v2;

	//Undistored
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
		CV_16SC2, map1, map2);

	for (int i = 0; i < imageNum; i++)
	{
		remap(views[i], v2, map1, map2, INTER_LINEAR);
		imshow("Undistored", v2);
		waitKey(100);
	}

	cout << "RMS: " << rms << endl;

	return rms;
}

void ch_undistort(InputArray src, OutputArray dst, InputArray map1, InputArray map2)
{
	remap(src, dst, map1, map2, INTER_LINEAR);
}

static void calObjectPosition(Size boardSize, float squareSize, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
}

static vector<vector<Point3f>> ch_calObjectCorners(Size boardSize, float squqreSize, vector<vector<Point2f>> imageCorners)
{
	vector<vector<Point3f>> corners(1);
	calObjectPosition(boardSize, squqreSize, corners[0]);
	corners.resize(imageCorners.size(), corners[0]);
	return corners;
}