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
	Mat_<float> gradDir(src_grey.rows, src_grey.cols);

  magnitude(grad_x, grad_y, gradMag);


  for(int y = 0; y < gradDir.rows; y++){
	for(int x = 0; x < src_grey.cols; x++){
		gradDir.at<float>(y,x) = atan(grad_y.at<float>(y,x) / grad_x.at<float>(y,x));
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
  printf("WTF\n");

  int thetaMin = -90;
  int thetaMax = 90;
  int thetaGrain = 1;

  int edge_thr = 100;

  int rhoMax = (int) gradMag.cols;
  printf("!!!!!! %d !!!!!!!!\n", rhoMax);

  int dims[2] = {rhoMax, (int) (thetaMax - thetaMin)/thetaGrain};
  Mat_<int> hough_space(2, dims, CV_8UC1);

  for(int y = 0; y < gradMag.rows; y++){ //Go through every pixel in the image.
	for(int x = 0; x < gradMag.cols; x++){
		if(gradMag.at<float>(y,x) == 255){ //Check it is an edge
			float theta = gradDir.at<float>(y,x);
			for(int thetaOff = -10; thetaOff <= 10; thetaOff++){ //if it is add line
				float newThetaRads = theta + ((thetaOff * 2 * CV_PI)/360);
				int newThetaDegs = (int) ((newThetaRads * 360) /(2 * CV_PI));
				if((newThetaDegs % thetaGrain) > -5 && (newThetaDegs % thetaGrain) < 5 ){ //check the value is at a sampled angle
					int rho = (int) x * cos(newThetaRads) + y * sin(newThetaRads);
					if(rho > 0 && rho < rhoMax){
						//printf("Is %d < %d\n",rho, rhoMax);
						hough_space.at<int>(rho, (newThetaDegs - thetaMin) / thetaGrain)++;
					}
				}
			}
		}
	}
  }

	Mat lineImage = src.clone();
	Mat pointImage = src.clone();
	int vote_thresh = 50;

	for(int theta1 = 0; theta1 < hough_space.size[1]; theta1++){
		for(int rho1 = 0; rho1 < hough_space.size[0]; rho1++){
			if(hough_space.at<int>(rho1, theta1) > vote_thresh){
				int x0 = 0;
				int y0 = 0;
				int x1 = 100;
				int y1 = 100;
				if(x0 >= 0 && x0 < gradMag.cols && y0 >= 0 && y0 < gradMag.rows){
					line(lineImage, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 255), 1);
				}
			}
			else{
				hough_space.at<int>(rho1, theta1) = 0;
			}
		}
	}
	printf("hough space dims: %d, %d\n", hough_space.size[0], hough_space.size[1]);

	for(int theta1 = 0; theta1 < hough_space.size[1]; theta1++){
		for(int rho1 = 0; rho1 < hough_space.size[0]; rho1++){
			int rhoMin = 0;
			if(hough_space.at<int>(rho1,theta1) > 100)
			for(int theta2 = 0; theta2 < hough_space.size[1]; theta2++){
				for(int rho2 = 0; rho2 < hough_space.size[0]; rho2++){
					//printf("Theta 1: %d\n, Theta2: %d, Rho1: %d, Rho2: %d\n", theta1, theta2, rho1, rho2);
					if(theta1 != theta2){                       //removes parallel lines
						if(rho1 != rho2){ //removes self reference
							if(hough_space.at<int>(rho2,theta2) > 100){
								y = (rho1 - rho2);
								int ySoSrs = (int)(yNum/(float) yDen);
								int xElent = (int)(xNum/(float) xDen);
								printf("1. Test at t1: %d, t2  %d, rho1 %d, rho2 %d\n", theta1, theta2, rho1, rho2);
								printf("a1: %d, a2 %d, rho1: %d, rho2: %d\n", a1, a2, rho1, rho2);
								printf("a1': %d",a1P);
								printf("xNum: %d, xDen: %d, x: %d\n", xNum, xDen, xElent);
								//std::cout << ("xNum: %d, xDen: %d, x: %d\n", xNum, xDen, xElent) << std::endl;
								//printf("yNum: %d, yDen: %d, y: %d\n", yNum, yDen, ySoSrs);
								printf("yNum: %d, yDen: %d, y: %d\n", yNum, yDen, ySoSrs);

								printf("Intersection at %d, %d\n", xElent, ySoSrs);

								circle(pointImage, Point(xElent ,ySoSrs), 2, Scalar(255,255,0), 1);
							}
						}
					}
				}
			rhoMin++;
			}
		}
	}

	imwrite("HoughSpace.png", hough_space);
	imwrite("points.png", pointImage);
	imwrite("lines.png", lineImage);

	//TODO Generalize, scales & ovals & stuff
}
