// TrackerTest.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"


#define NEW_ULSEE_TRACKER


#ifdef NEW_ULSEE_TRACKER

#include "libTracker.h"
#else

#include "libTracker_old.h"
#endif


#include <opencv2/highgui/highgui.hpp>


#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <thread>
#include <mutex>

using namespace boost::interprocess;


const std::string kActivationKey = "e9f7xxphqp7F9TWcWJCouzFpbfQozp1I";





class FXModel {
public:


	bool inited;
	int pointId;
	int pointTotal;
	std::array<float, 66 * 2> pts;
	float viewerSize;
	float faceWidth;
	float faceWidthRaw;
	float xFaceCenter;
	float yFaceCenter;
	float xTopLipCenter;
	float yTopLipCenter;
	int pitch;
	int yaw;
	int roll;
	float xFaceCenterRaw;
	float yFaceCenterRaw;
	float pupilLX;
	float pupilLY;
	float pupilRX;
	float pupilRY;
	float gazeLX;
	float gazeLY;
	float gazeRX;
	float gazeRY;
	float gazeLZ;
	float gazeRZ;
	int fWidth;
	int fHeight;
	int width;
	int height;
	float widthDiff;

	std::array<float, 66 * 2> pts3d;

	std::array<float, 66> confidence;

	cv::Rect faceRect;

	float xTopLipCenterFixed;
	float yTopLipCenterFixed;
};


//=============================================================================
//function to draw points with antialiasing
void drawPoint(const float xc, const float yc, const float rad,
	cv::Mat &I, const cv::Scalar &color, const bool full) {
	int x0 = cv::min(I.cols - 1, cv::max(0, static_cast<int>(floor(xc - rad))));
	int x1 = cv::min(I.cols - 1, cv::max(0, static_cast<int>(ceil(xc + rad))));
	int y0 = cv::min(I.rows - 1, cv::max(0, static_cast<int>(floor(yc - rad))));
	int y1 = cv::min(I.rows - 1, cv::max(0, static_cast<int>(ceil(yc + rad))));
	cv::Vec3b colorB(color[0], color[1], color[2]);
	if (!full) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				float vx = x - xc, vy = y - yc;
				float v = fabsf(sqrtf(vx * vx + vy * vy) - rad);
				if (v <= 1.0f) {
					float w = 1.0f - v;
					I.at<cv::Vec3b>(y, x) = v * I.at<cv::Vec3b>(y, x) + w * colorB;
				}
			}
		}
	}
	else {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				float vx = x - xc, vy = y - yc;
				float v = sqrtf(vx * vx + vy * vy) - rad;
				if (v <= 0) {
					I.at<cv::Vec3b>(y, x) = colorB;
				}
				else if (v <= 1.0f) {
					float w = 1.0f - v;
					I.at<cv::Vec3b>(y, x) = v * I.at<cv::Vec3b>(y, x) + w * colorB;
				}
			}
		}
	}
}
//=============================================================================
//function to draw points with antialiasing
void drawPointAA(const float xc, const float yc, const float rad,
	cv::Mat &I, const cv::Scalar &color, const bool full) {
	int x0 = cv::min(I.cols - 1, cv::max(0, static_cast<int>(floor(xc - rad))));
	int x1 = cv::min(I.cols - 1, cv::max(0, static_cast<int>(ceil(xc + rad))));
	int y0 = cv::min(I.rows - 1, cv::max(0, static_cast<int>(floor(yc - rad))));
	int y1 = cv::min(I.rows - 1, cv::max(0, static_cast<int>(ceil(yc + rad))));
	cv::Vec3b colorB(color[0], color[1], color[2]);
	if (!full) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				float vx = x - xc, vy = y - yc;
				float v = fabsf(sqrtf(vx * vx + vy * vy) - rad);
				if (v <= 1.0f) {
					float w = 1.0f - v;
					I.at<cv::Vec3b>(y, x) = v * I.at<cv::Vec3b>(y, x) + w * colorB;
				}
			}
		}
	}
	else {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				float vx = x - xc, vy = y - yc;
				float v = sqrtf(vx * vx + vy * vy) - rad;
				if (v <= 0) {
					I.at<cv::Vec3b>(y, x) = colorB;
				}
				else if (v <= 1.0f) {
					float w = 1.0f - v;
					I.at<cv::Vec3b>(y, x) = v * I.at<cv::Vec3b>(y, x) + w * colorB;
				}
			}
		}
	}
}
class aa_draw {
public:
	static void plot(int x, int y, float w, cv::Mat &I, const cv::Scalar &color,
		const float alpha = 0.5f) {

		if (w > 1) {
			std::cout << "too large: " << w << std::endl; exit(0);
		}
		float wa = w * alpha;
		int c = I.cols; float v = 1.0f - wa;
		cv::Vec3b colorB(color[0], color[1], color[2]);
		for (int k = 0; k < 3; k++) {
			I.at<cv::Vec3b>(y, x) = v * I.at<cv::Vec3b>(y, x) + wa * colorB;
		}
	}
	static int ipart(float x) { return static_cast<int>(floor(x)); }
	static int round(float x) { return aa_draw::ipart(x + 0.5f); }
	static float fpart(float x) { return x - aa_draw::ipart(x); }
	static float rfpart(float x) { return 1.0f - aa_draw::fpart(x); }
	static void drawLine(float x0, float y0, float x1, float y1,
		cv::Mat &I, const cv::Scalar &c) {
		if ((x0 < 0) || (x0 >= I.cols) ||
			(x1 < 0) || (x1 >= I.cols) ||
			(y0 < 0) || (y0 >= I.rows) ||
			(y1 < 0) || (y1 >= I.rows)) {
			return;
		}

		bool steep = fabs(y1 - y0) > fabs(x1 - x0);
		if (steep) {
			{ float v = x0; x0 = y0; y0 = v; }
			{ float v = x1; x1 = y1; y1 = v; }
		}
		if (x0 > x1) {
			{ float v = x0; x0 = x1; x1 = v; }
			{ float v = y0; y0 = y1; y1 = v; }
		}
		float dx = x1 - x0;
		float dy = y1 - y0;
		float gradient = dy / dx;

		// handle first endpoint
		int xend = aa_draw::round(x0);
		float yend = y0 + gradient * (xend - x0);
		float xgap = aa_draw::rfpart(x0 + 0.5f);
		int xpxl1 = xend;   //this will be used in the main loop
		int ypxl1 = aa_draw::ipart(yend);
		if (steep) {
			aa_draw::plot(ypxl1, xpxl1, aa_draw::rfpart(yend) * xgap, I, c);
			aa_draw::plot(ypxl1 + 1, xpxl1, aa_draw::fpart(yend) * xgap, I, c);
		}
		else {
			aa_draw::plot(xpxl1, ypxl1, aa_draw::rfpart(yend) * xgap, I, c);
			aa_draw::plot(xpxl1, ypxl1 + 1, aa_draw::fpart(yend) * xgap, I, c);
		}
		float intery = yend + gradient; // first y-intersection for the main loop

										// handle second endpoint
		xend = aa_draw::round(x1);
		yend = y1 + gradient * (xend - x1);
		xgap = aa_draw::fpart(x1 + 0.5f);
		int xpxl2 = xend; //this will be used in the main loop
		int ypxl2 = aa_draw::ipart(yend);
		if (steep) {
			aa_draw::plot(ypxl2, xpxl2, aa_draw::rfpart(yend) * xgap, I, c);
			aa_draw::plot(ypxl2 + 1, xpxl2, aa_draw::fpart(yend) * xgap, I, c);
		}
		else {
			aa_draw::plot(xpxl2, ypxl2, aa_draw::rfpart(yend) * xgap, I, c);
			aa_draw::plot(xpxl2, ypxl2 + 1, aa_draw::fpart(yend) * xgap, I, c);
		}
		// main loop
		for (int x = xpxl1 + 1; x <= xpxl2 - 1; x++) {
			if (steep) {
				aa_draw::plot(aa_draw::ipart(intery), x, aa_draw::rfpart(intery),
					I, c);
				aa_draw::plot(aa_draw::ipart(intery) + 1, x, aa_draw::fpart(intery),
					I, c);
			}
			else {
				aa_draw::plot(x, aa_draw::ipart(intery), aa_draw::rfpart(intery),
					I, c);
				aa_draw::plot(x, aa_draw::ipart(intery) + 1, aa_draw::fpart(intery),
					I, c);
			}
			intery = intery + gradient;
		}
	}
};
//=============================================================================




shared_memory_object sharedMemory;
mapped_region switcher_region;
mapped_region flag_region;
mapped_region width_region;
mapped_region height_region;
mapped_region image_region;

mapped_region output_image_region;
mapped_region fxmodel_region;



int w = 1080;
int h = 1920;


cv::Mat frame(h, w, CV_8UC3);
cv::Mat img(h, w, CV_8UC3);


volatile int c;

std::mutex frameMutex;




namespace FaceTracker {
	const static int MAX_TO_TRACK = 5;

	// if you add more tracker values, you must update this number with the total amount values.
	const static int NUMBER_OF_PRESET_VALUES = 31; // not including raw points

	const static int FACE_CENTER_X = 0;
	const static int FACE_CENTER_Y = 1;
	const static int PITCH = 2;
	const static int YAW = 3;
	const static int ROLL = 4;
	const static int LEFT_PUPIL_X = 5;
	const static int LEFT_PUPIL_Y = 6;
	const static int RIGHT_PUPIL_X = 7;
	const static int RIGHT_PUPIL_Y = 8;
	const static int LEFT_GAZE_X = 9;
	const static int LEFT_GAZE_Y = 10;
	const static int RIGHT_GAZE_X = 11;
	const static int RIGHT_GAZE_Y = 12;
	const static int FACE_WIDTH = 13;
	const static int POINT_ID = 14;
	const static int POINT_TOTAL = 15;
	const static int TOP_LIP_CENTER_X = 16;
	const static int TOP_LIP_CENTER_Y = 17;
	// Functionality: Get the 3D position of head center in world coordinate -- see doc
	const static int HEAD_3D_POS_X = 18;
	const static int HEAD_3D_POS_Y = 19;
	const static int HEAD_3D_POS_Z = 20;
	//Functionality: Get the 3D position of head center in camera coordinate -- see doc
	const static int HEAD_3D_POS_CC_X = 21;
	const static int HEAD_3D_POS_CC_Y = 22;
	const static int HEAD_3D_POS_CC_Z = 23;
	const static int FACE_SCALE = 24;

	const static int FACE_RECT_X = 25;
	const static int FACE_RECT_Y = 26;
	const static int FACE_RECT_WIDTH = 27;
	const static int FACE_RECT_HEIGHT = 28;

	const static int LEFT_GAZE_Z = 29;
	const static int RIGHT_GAZE_Z = 30;

	// different set of presets, revolving around the const float*
	const static int RAW_POINTS = 0;
	const static int RAW_3D_POINTS = 1;

}

std::array<FXModel, 5> fxModelArr;

void CreateFxModelArrFromPoints(int frameCols, int frameRows, const float* rawPoints[5][2], float trackerValues[5][FaceTracker::NUMBER_OF_PRESET_VALUES], const float* confidence[FaceTracker::MAX_TO_TRACK])
{


	for (int id = 0; id < 5; id++)
	{

		fxModelArr[id].inited = false;

		if (trackerValues[id][FaceTracker::POINT_ID] >= 0)
		{

			float imgWidth = frameCols;
			float imgHeight = frameRows;
			float widthDiff = 1.0f;
			float heightDiff = 1.0f;

			if (imgWidth != w) {
				widthDiff = WAIT_CHILD / imgWidth;
			}

			if (imgHeight != h) {
				heightDiff = h / imgHeight;
			}

			// calculate x and y face center values.
			float xValue = (float)-(1 - ((trackerValues[id][FaceTracker::FACE_CENTER_X]) / (imgWidth / 2)));
			float yValue = (float)(1 - ((trackerValues[id][FaceTracker::FACE_CENTER_Y]) / (imgHeight / 2)));

			// calculate x and y face center values.
			float lipsXValue = (float)-(1 - ((trackerValues[id][FaceTracker::TOP_LIP_CENTER_X]) / (imgWidth / 2)));
			float lipsYValue = (float)(1 - ((trackerValues[id][FaceTracker::TOP_LIP_CENTER_Y]) / (imgHeight / 2)));

			float facewidth = trackerValues[id][FaceTracker::FACE_WIDTH] / imgWidth;

			float viewerSize = (facewidth * 2) * widthDiff;

			int pitch = (int)trackerValues[id][FaceTracker::PITCH];
			int yaw = 0 - (int)trackerValues[id][FaceTracker::YAW];
			int roll = 0 - (int)trackerValues[id][FaceTracker::ROLL];

			FXModel fxModel = FXModel();
			fxModel.pointId = id;
			fxModel.pointTotal = trackerValues[id][FaceTracker::POINT_TOTAL];
			std::copy(&rawPoints[id][FaceTracker::RAW_POINTS][0], &rawPoints[id][FaceTracker::RAW_POINTS][66 * 2], &fxModel.pts[0]);
			std::copy(&rawPoints[id][FaceTracker::RAW_POINTS][0], &rawPoints[id][FaceTracker::RAW_POINTS][66 * 3], &fxModel.pts3d[0]);
			//fxModel.pts = rawPoints[id][FaceTracker::RAW_POINTS];
			//fxModel.pts3d = rawPoints[id][FaceTracker::RAW_3D_POINTS];
			fxModel.faceWidth = facewidth;
			fxModel.widthDiff = widthDiff;
			fxModel.faceWidthRaw = trackerValues[id][FaceTracker::FACE_WIDTH];
			fxModel.xFaceCenter = xValue;
			fxModel.yFaceCenter = yValue;
			fxModel.xTopLipCenter = trackerValues[id][FaceTracker::TOP_LIP_CENTER_X];
			fxModel.yTopLipCenter = trackerValues[id][FaceTracker::TOP_LIP_CENTER_Y];
			fxModel.xFaceCenterRaw = trackerValues[id][FaceTracker::FACE_CENTER_X];
			fxModel.yFaceCenterRaw = trackerValues[id][FaceTracker::FACE_CENTER_Y];
			fxModel.pitch = pitch;
			fxModel.roll = roll;
			fxModel.yaw = yaw;
			fxModel.viewerSize = viewerSize;
			fxModel.gazeLX = trackerValues[id][FaceTracker::LEFT_GAZE_X];
			fxModel.gazeLY = trackerValues[id][FaceTracker::LEFT_GAZE_Y];
			fxModel.gazeRX = trackerValues[id][FaceTracker::RIGHT_GAZE_X];
			fxModel.gazeRY = trackerValues[id][FaceTracker::RIGHT_GAZE_Y];
			fxModel.gazeRZ = trackerValues[id][FaceTracker::RIGHT_GAZE_Z];
			fxModel.gazeLZ = trackerValues[id][FaceTracker::LEFT_GAZE_Z];
			fxModel.pupilLX = trackerValues[id][FaceTracker::LEFT_PUPIL_X];
			fxModel.pupilLY = trackerValues[id][FaceTracker::LEFT_PUPIL_Y];
			fxModel.pupilRX = trackerValues[id][FaceTracker::RIGHT_PUPIL_X];
			fxModel.pupilRY = trackerValues[id][FaceTracker::RIGHT_PUPIL_Y];
			fxModel.fWidth = frameCols;
			fxModel.fHeight = frameRows;
			fxModel.width = w;
			fxModel.height = h;
			fxModel.faceRect = cv::Rect(trackerValues[id][FaceTracker::FACE_RECT_X], trackerValues[id][FaceTracker::FACE_RECT_Y], trackerValues[id][FaceTracker::FACE_RECT_WIDTH], trackerValues[id][FaceTracker::FACE_RECT_HEIGHT]);
			/*
			for (size_t i = 0; i < 66; i++)
			{
			fxModel.confidence[i] = confidence[id][i];
			}*/

			fxModel.xTopLipCenterFixed = lipsXValue;
			fxModel.yTopLipCenterFixed = lipsYValue;

			fxModel.inited = true;

			fxModelArr[id] = fxModel;
		}


	}
}


double distanceBetween(double x1, double y1, double x2, double y2)
{
	//calculating number to square in next step
	double x = x1 - x2;
	double y = y1 - y2;
	double dist;

	//calculating Euclidean distance
	dist = pow(x, 2) + pow(y, 2);
	dist = sqrt(dist);

	return dist;
}




void innerProcess()
{
	bool exit = false;

	int width;
	int height;


	while (!exit)
	{
		memcpy(&width, width_region.get_address(), sizeof(int));
		memcpy(&height, height_region.get_address(), sizeof(int));

		if (width <= 0)
		{
			width = 4;
		}

		if (height <= 0)
		{
			height = 4;
		}


		if (frame.cols != width || frame.rows != height)
		{
			frame = cv::Mat(height, width, CV_8UC3);
		}


		memcpy(frame.data, image_region.get_address(), width*height * 3);


		int max_track = 5;

		int rtnstate;

		float values[FaceTracker::MAX_TO_TRACK][FaceTracker::NUMBER_OF_PRESET_VALUES];
		const float* rawPoints[FaceTracker::MAX_TO_TRACK][2];
		const float* confidence[FaceTracker::MAX_TO_TRACK];


		using namespace FaceTracker;

#ifdef NEW_ULSEE_TRACKER

		rtnstate = ULS_TrackerProcessByte((void*)img.data, img.cols, img.rows, 77, 0, NULL);

#else

		float yawAngle;

		rtnstate = ULS_TrackerProcessByte((void*)frame.data, frame.cols, frame.rows, 1, 0, &yawAngle);
#endif
		if (rtnstate < 0) {
			//qDebug() << "state:" << rtnstate;
			values[0][0] = -1.0f;
			return;
		}
		else if (rtnstate >= 0) {
			int n = ULS_GetTrackerPointNum();
			for (int j = 0; j < MAX_TO_TRACK; j++) {

				//	rawPoints[j] = NULL;
				const float* pts = ULS_GetTrackerPoint(j);

				const float* pts3d = ULS_GetTrackerPoint3D(j);

				if (pts == NULL) {
					values[j][POINT_ID] = -1.0f;
					continue;
				}

				int fx, fy, fw, fh;

				bool faceRectResult = ULS_GetTrackerFacerect(j, fx, fy, fw, fh);

				if (!faceRectResult)
				{
					values[j][POINT_ID] = -1.0f;
					continue;
				}

				// face center
				float imx, imy;
				ULS_GetFaceCenter(imx, imy, j);

#if 0
				// without pose stability
				float pitch = ULS_GetPitchRadians(j);
				float yaw = ULS_GetYawRadians(j);
				float roll = ULS_GetRollRadians(j);
#else
				// with pose stability
				float pitch = ULS_GetStablePitchRadians(j);
				float yaw = ULS_GetStableYawRadians(j);
				float roll = ULS_GetStableRollRadians(j);
#endif			
				cv::Point2f pupils[2];
				ULS_GetLeftPupilXY(pupils[0].x, pupils[0].y, j);
				ULS_GetRightPupilXY(pupils[1].x, pupils[1].y, j);

				cv::Point3f gazes[2];
				ULS_GetLeftGaze(gazes[0].x, gazes[0].y, gazes[0].z, j);
				ULS_GetRightGaze(gazes[1].x, gazes[1].y, gazes[1].z, j);


				// project head center in world coordinate to image plane
				//	float nx, ny;
				float hx, hy, hz;
				ULS_GetHead3DPosition(hx, hy, hz, j);

				float hxCC, hyCC, hzCC;
				ULS_GetHead3DPositionCC(hxCC, hyCC, hzCC, j);

				float faceScale;
				faceScale = ULS_GetScaleInImage(j);



				int lX = pts[0];
				int ly = pts[1];
				int rX = pts[32];
				int ry = pts[33];

				// #1 = pts[2 * i], #2 = pts[2 * i + 1]
				// point # 28
				int noseX = pts[56];
				int noseY = pts[57];

				int lipX = pts[102];
				int lipY = pts[103];

				values[j][TOP_LIP_CENTER_X] = lipX;
				values[j][TOP_LIP_CENTER_Y] = lipY;

				values[j][FACE_CENTER_X] = noseX;
				values[j][FACE_CENTER_Y] = noseY;
				values[j][PITCH] = pitch;
				values[j][YAW] = yaw;
				values[j][ROLL] = roll;
				values[j][LEFT_PUPIL_X] = pupils[0].x;
				values[j][LEFT_PUPIL_Y] = pupils[0].y;
				values[j][RIGHT_PUPIL_X] = pupils[1].x;
				values[j][RIGHT_PUPIL_Y] = pupils[1].y;
				values[j][LEFT_GAZE_X] = gazes[0].x;
				values[j][LEFT_GAZE_Y] = gazes[0].y;
				values[j][LEFT_GAZE_Z] = gazes[0].z;
				values[j][RIGHT_GAZE_X] = gazes[1].x;
				values[j][RIGHT_GAZE_Y] = gazes[1].y;
				values[j][RIGHT_GAZE_Z] = gazes[1].z;
				values[j][FACE_WIDTH] = distanceBetween(int(pts[0]), int(pts[1]), int(pts[32]), int(pts[33]));;
				values[j][POINT_ID] = (float)j;
				values[j][POINT_TOTAL] = (float)n;
				values[j][HEAD_3D_POS_X] = hx;
				values[j][HEAD_3D_POS_Y] = hy;
				values[j][HEAD_3D_POS_Z] = hz;
				values[j][HEAD_3D_POS_CC_X] = hxCC;
				values[j][HEAD_3D_POS_CC_Y] = hyCC;
				values[j][HEAD_3D_POS_CC_Z] = hzCC;
				values[j][FACE_SCALE] = faceScale;

				values[j][FACE_RECT_X] = fx;
				values[j][FACE_RECT_Y] = fy;

				values[j][FACE_RECT_WIDTH] = fw;
				values[j][FACE_RECT_HEIGHT] = fh;

				// rawPoints set--
				rawPoints[j][RAW_POINTS] = pts;
				rawPoints[j][RAW_3D_POINTS] = pts3d;


			}

		}



		CreateFxModelArrFromPoints(frame.cols, frame.rows, rawPoints, values, confidence);



		memcpy(fxmodel_region.get_address(), &fxModelArr[0], 5 * sizeof(FXModel));

		memcpy(output_image_region.get_address(), &frame.data[0], frame.cols*frame.rows * 3);

		char flag = 1;

		memcpy(flag_region.get_address(), &flag, 1);


		//std::unique_lock<std::mutex> lock(frameMutex);

		frame.copyTo(img);

		char switcher = 0;

		memcpy(&switcher, switcher_region.get_address(), 1);


		if (switcher == 1)
		{
			exit = true;
		}

		/*cv::imshow("ULS_Tracker", img);
		/c = cv::waitKey(5);

		if (c == 27)
		{
			exit = true;
		}
		*/

	}
}


int main(int argc, char * argv[])
{

	std::cout << "ULS face tracker\n"
		<< "(c) 2017 ULSee Inc. \n"
		<< "All rights reserved\n"
		<< "www.ulsee.com\n\n"
		<< "Key control: [esc]: quit\n\n";

	std::string key;
	if (argc > 1) {
		key = argv[1];
	}
	else {
		key = kActivationKey;
	}

	//Init Tracker
	int max_tracked_face = 5;


#ifdef NEW_ULSEE_TRACKER
	int rtnValue = ULS_TrackerInit("./model/", (char*)key.c_str(), max_tracked_face, 0, 200, 2);
#else
	int rtnValue = ULS_TrackerInit("./model/", const_cast<char*>(key.c_str()), 5, 0, 40);
#endif

	if (rtnValue <= 0) {
		ULS_TrackerFree();

		return -1;
	}


	int max_track = max_tracked_face;

	cv::Mat_<int> connections =
		(cv::Mat_<int>(61, 2) <<
			0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
			10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18,
			18, 19, 19, 20, 20, 21, 22, 23, 23, 24, 24, 25, 25, 26,
			27, 28, 28, 29, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35,
			36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 36, 42, 43,
			43, 44, 44, 45, 45, 46, 46, 47, 47, 42, 48, 49, 49, 50,
			50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57,
			57, 58, 58, 59, 59, 48, 60, 65, 60, 61, 61, 62, 62, 63,
			63, 64, 64, 65);
	std::vector<cv::Scalar> colors(max_track); cv::RNG rn;
	for (int i = 0; i < max_track; i++) {
		if (i == 0) { colors[i] = CV_RGB(255, 0, 0); }
		else if (i == 1) { colors[i] = CV_RGB(0, 255, 0); }
		else if (i == 2) { colors[i] = CV_RGB(0, 0, 255); }
		else if (i == 3) { colors[i] = CV_RGB(255, 0, 255); }
		else if (i == 4) { colors[i] = CV_RGB(255, 255, 0); }
		else if (i == 5) { colors[i] = CV_RGB(0, 255, 255); }
		else {
			colors[i] =
				CV_RGB(rn.uniform(0, 255), rn.uniform(0, 255), rn.uniform(0, 255));
		}
	}

	cv::namedWindow("ULS_Tracker");
	cv::resizeWindow("ULS_Tracker", 640, 480);

#ifdef NEW_ULSEE_TRACKER
	ULS_SetSmooth(true);
	ULS_SetStablePose(false, 3);
	ULS_SetTwoThresholds(0.25, 0.27);
#else


	ULS_SetSmooth(false);
	ULS_SetTwoThresholds(0.27f, 0.33f);
	ULS_SetTwoThresholds(0.25f, 0.28f);
#endif
	cv::Mat im(h, w, CV_8UC1);
	cv::Mat dwnimg(h, w, CV_8UC1);
	cv::Mat smallImg(h, w, CV_8UC1);

	int rtnstate = 0;
	int iter;
	int fx, fy, fw, fh;
	unsigned char *pGray = new unsigned char[w * h];


	//std::cout << "Test 1" << std::endl;

	sharedMemory = shared_memory_object(open_only, "test_shared_memory", read_write);
	//std::cout << "Test 2" << std::endl;
	switcher_region = mapped_region(sharedMemory, read_only, 0, 1);
	//std::cout << "Test 3" << std::endl;
	flag_region = mapped_region(sharedMemory, read_write, 1, 1);

	//std::cout << "Test 4" << std::endl;
	width_region = mapped_region(sharedMemory, read_only, 2, sizeof(int));
	height_region = mapped_region(sharedMemory, read_only, 2 + sizeof(int), sizeof(int));

	image_region = mapped_region(sharedMemory, read_only, 2 + 2 * sizeof(int), w * h * 3);

	//std::cout << "Test 5" << std::endl;

	fxmodel_region = mapped_region(sharedMemory, read_write, 2 + 2 * sizeof(int) + w * h * 3, sizeof(FXModel) * 5);

	output_image_region = mapped_region(sharedMemory, read_write, 2 + 2 * sizeof(int) + w * h * 3 + sizeof(FXModel) * 5, w * h * 3);

	//std::cout << "Test 6" << std::endl;

	innerProcess();
	

	//std::cout << "Test 7" << std::endl;

	delete[]pGray;

	ULS_TrackerFree();

	return 0;
}
