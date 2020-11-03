// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat src_grey;

  /**
  *** Sobel edge detection code example courtesy of openCV tutorial
  *** https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
  **/

  GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT);

  cvtColor(src, src_grey, CV_BGR2GRAY);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(src_grey, grad_x, CV_32F, 1, 0, 3);
  Sobel(src_grey, grad_y, CV_32F, 0, 1, 3);

  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);

  Mat gradMag;
	Mat gradDir = src_grey.clone();

  magnitude(grad_x, grad_y, gradMag);


	for(int y = 0; y < gradDir.rows; y++){
		for(int x = 0; x < src_grey.cols; x++){
			gradDir.at<uchar>(y,x) = atan(grad_y.at<float>(y,x) / grad_x.at<float>(y,x));
		}
	}


	int thr = 100;
  for(int i = 0; i < gradMag.rows; i++)
  {
    for(int j = 0; j < gradMag.cols; j++)
    {
      if(gradMag.at<float>(i,j) < thr) gradMag.at<float>(i,j) = 0;
      else gradMag.at<float>(i,j) = 255;
    }
  }

  imwrite("grad.png", gradMag);
	imwrite("dir.png", gradDir);


	int radmin = 30;
	int radmax = 50;
	int radGrain = 1;

	//float yScaleMin = 1; Has to be y to keep hough_space as ints
	//float yScaleMax = 1; can change it to x to make it easier to visualise but
	//float yScaleGrain = 10; it might be more difficult to think about codewise

	//float rotMin = 0; keep in degrees and change in sin/cos to rads to keep
	//float rotMax = 360; hough space as ints
	//float rotGrain = 1;

	int edge_thr = 100;

	int dims[3] = {gradMag.rows, gradMag.cols, radmax-radmin};
	//int dims[5] = {gradMag.rows, gradMag.cols, radmax-radmin, yScaleMax - yScaleMin, rotationMax-rotationMin};
	Mat_<int> hough_space(3, dims, CV_8UC1);

	for(int y = 0; y < gradMag.rows; y++){ //Go through every pixel in the image.
		for(int x = 0; x < gradMag.cols; x++){
			if(gradMag.at<float>(y,x) == 255){
				for(int r = radmin; r < radmax; r += radGrain){ //Go through all values of r
					//for(int yScale = yScaleMin; yScale <= yScaleMax; yScale += yScaleGrain){}
						//for(int rot = rotMin; rot <= rotMax; rot += rotGrain){}
							//int x_0 = x - r*cos(gradDir.at<uchar>(y,x) + (rot * 2pi/360))
							//int y_0 = y*yScale - r * sin(gradDir.at<uchar>(y,x) + (rot * 2pi/360))
							//if(x_0 > 0 && x_0 < gradMag.cols && y_0 > 0 && y_0 < gradMag.rows)
								//hough_space.at<int>(y_0, x_0, r-radmin, yScale - yScaleMin, rot)
					int x_0 = x - r * cos(gradDir.at<uchar>(y,x)); //Vote on
					int y_0 = y - r * sin(gradDir.at<uchar>(y,x));
					if(x_0 > 0 && x_0 < gradMag.cols && y_0 > 0 && y_0 < gradMag.rows){
						hough_space.at<int>(y_0, x_0, r-radmin) += 1;
					}

					x_0 = x + r * cos(gradDir.at<uchar>(y,x));
					y_0 = y + r * sin(gradDir.at<uchar>(y,x));
					if(x_0 > 0 && x_0 < gradMag.cols && y_0 > 0 && y_0 < gradMag.rows){
						hough_space.at<int>(y_0, x_0, r-radmin) += 1;
					}
				}
			}
		}
	}

	Mat circleImage = src.clone();

	int vote_thresh = 3;

	for(int y = 0; y < hough_space.size[0]; y++){
		for(int x = 0; x < hough_space.size[1]; x++){
			for(int r = 0; r < hough_space.size[2]; r++){
				if(hough_space.at<int>(y, x, r) > vote_thresh){
					circle(circleImage, Point(x, y), r+radmin, Scalar(0, 255, 255), 0.5);
				}
			}
		}
	}
	imwrite("circles.png", circleImage);

	//TODO Generalize, scales & ovals & stuff
}
