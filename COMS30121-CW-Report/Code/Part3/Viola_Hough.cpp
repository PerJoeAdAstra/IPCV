/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void HoughDetectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
  HoughDetectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

void HoughDetectAndDisplay( Mat src ){
  std::vector<Rect> dartBoards;
	Mat src_grey;

  // 1. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor(src, src_grey, CV_BGR2GRAY);
	equalizeHist( src_grey, src_grey );

  // 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( src_grey, dartBoards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of dartBoards found
  std::cout << dartBoards.size() << std::endl;

 /* Commented out code for drawing dartboards detected by Viola Jones*/ 
 //  for( int i = 0; i < dartBoards.size(); i++ )
	// {
	// 	rectangle(src, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 255, 0, 0 ), 2);
	// }

  // 4. Do Hough transform for dartboards
  Mat grad_x, grad_y;

  Sobel(src_grey, grad_x, CV_32F, 1, 0, 3);
  Sobel(src_grey, grad_y, CV_32F, 0, 1, 3);

  Mat gradMag;
	Mat_<float> gradDir(src_grey.rows, src_grey.cols);

  magnitude(grad_x, grad_y, gradMag);         //creating mag image
	for(int y = 0; y < gradDir.rows; y++){      //creating dir image
		for(int x = 0; x < src_grey.cols; x++){
			gradDir.at<float>(y,x) = atan(grad_y.at<float>(y,x) / grad_x.at<float>(y,x));
		}
	}

  int thr = 100;                              //Threshold grad image
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

	int radmin = 20;
	int radmax = 100;
	int radGrain = 1;

  //TODO: Detect intersecting lines (probably easier the ellipses)
  int dims[3] = {gradMag.rows, gradMag.cols, radmax-radmin};
	Mat_<int> hough_space(3, dims, CV_8UC1);
	for(int y = 0; y < gradMag.rows; y++){ //Go through every pixel in the image.
		for(int x = 0; x < gradMag.cols; x++){
			if(gradMag.at<float>(y,x) == 255){
				for(int r = radmin; r < radmax; r += radGrain){ //Go through all values of r
					int x_0 = x - r * cos(gradDir.at<float>(y,x));
					int y_0 = y - r * sin(gradDir.at<float>(y,x));
					// If the centre is within the image
					if(x_0 > 0 && x_0 < gradMag.cols && y_0 > 0 && y_0 < gradMag.rows){
						hough_space.at<int>(y_0, x_0, r-radmin) += 1;
					}
					x_0 = x + r * cos(gradDir.at<float>(y,x));
					y_0 = y + r * sin(gradDir.at<float>(y,x));
					if(x_0 > 0 && x_0 < gradMag.cols && y_0 > 0 && y_0 < gradMag.rows){
						hough_space.at<int>(y_0, x_0, r-radmin) += 1;
					}
				}
			}
		}
	}

	Mat circleImage = src.clone();
	Mat_<int> flatHoughSpace(gradMag.rows, gradMag.cols);

	for(int y = 0; y < hough_space.size[0]; y++){
		for(int x = 0; x < hough_space.size[1]; x++){
			int votes = 0;
			for(int r = 0; r < hough_space.size[2]; r++){
				votes += hough_space.at<int>(y, x, r);
			}
			flatHoughSpace.at<int>(y, x) = votes;
		}
	}

  double minVal;
  double maxVal;
  minMaxLoc(flatHoughSpace, &minVal, &maxVal);

	int vote_thresh = (maxVal/hough_space.size[2])*5; //This threshold works okay
  // std::cout << vote_thresh << std::endl;

	for(int y = 0; y < hough_space.size[0]; y++){
		for(int x = 0; x < hough_space.size[1]; x++){
			for(int r = 0; r < hough_space.size[2]; r++){
				if(hough_space.at<int>(y, x, r) > vote_thresh){
					circle(circleImage, Point(x, y), r+radmin, Scalar(0, 255, 255), 0.5);
				}
			}
		}
	}
	imwrite("HoughSpace.png", flatHoughSpace);
	imwrite("circles.png", circleImage);

	//5. For Viola-Jones detected, check against hough space

	vector<Point> detected;
	vector<int> detectedRad;
  int detectedDartBoards = 0;

  for(int y = 0; y < hough_space.size[0]; y++){
		for(int x = 0; x < hough_space.size[1]; x++){
      int bestRad = -1;
      bool concentric = false;
			for(int r = 0; r < hough_space.size[2]; r++){
				if(hough_space.at<int>(y, x, r) > vote_thresh){
          if(hough_space.at<int>(y, x, r) >
             hough_space.at<int>(y, x, bestRad)){
            if(bestRad != -1){
              concentric = true;
              // circle(src, Point(x, y), bestRad + radmin, Scalar( 0, 255, 255 ), 0.5);
            }
            bestRad = r;
          }
        }
      }
      if(bestRad > -1 && concentric){
        detectedDartBoards++;
        detected.push_back(Point(x, y));
        detectedRad.push_back(bestRad);
  	    //circle(src, Point(x, y), bestRad + radmin, Scalar( 0, 0, 255 ), 2);
  		}
    }
  }

	for(int i = 0; i < dartBoards.size(); i++){
		int bestRad = -1;
		int x = dartBoards[i].x + (dartBoards[i].width /2);
		int y = dartBoards[i].y + (dartBoards[i].height /2);
		for(int r = 0; r < radmax-radmin; r++){
      for(int xOffset = -5; xOffset <= 5; xOffset++){
				for(int yOffset = -5; yOffset <=5; yOffset++){
					if(hough_space.at<int>(y + yOffset, x + xOffset, r) > vote_thresh){
						if(hough_space.at<int>(y + yOffset, x + xOffset, r) >
						 	 hough_space.at<int>(y + yOffset, x + xOffset, bestRad)){
							bestRad = r;
						}
	        }
		    }
      }
		}
		if(bestRad > -1){
			detectedDartBoards++;
			detected.push_back(Point(x,y));
			detectedRad.push_back(bestRad);
			// rectangle(src, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
	    //circle(src, Point(dartBoards[i].x + (dartBoards[i].width /2), dartBoards[i].y + (dartBoards[i].height/2)), bestRad + radmin, Scalar( 0, 0, 255 ), 2);
		}
	}

	// filter the detected to remove duplicates
	vector<Point> filteredDetected = detected;
	vector<int> filteredRad = detectedRad;
	int distanceThreshold = 50;
	for(int i = 0; i < detected.size(); i++){
		Point p = detected[i];
		for(int j = i+1; j < detected.size(); j++){
			Point pPrime = detected[j];
			if(abs(p.x - pPrime.x) < distanceThreshold && 
				 abs(p.y - pPrime.y) < distanceThreshold){
				int rad = detectedRad[j];
				filteredDetected.erase(remove(filteredDetected.begin(), filteredDetected.end(), pPrime), 
															 filteredDetected.end());
				filteredRad.erase(remove(filteredRad.begin(), filteredRad.end(), rad),
					                filteredRad.end());
			}
		}
	}

	// Draw the detected dartboards
	for(int i = 0; i < filteredDetected.size(); i++){
		int width = (filteredRad[i] + radmin) * 2;
		int height = width;
		int xLeft = filteredDetected[i].x - width/2;
		int xRight = xLeft + width;
		int yTop = filteredDetected[i].y - height/2;
		int yBottom = yTop + height;
		rectangle(src, Point(xLeft, yTop), Point(xRight, yBottom), Scalar( 0, 255, 0 ), 2);
	}
	// for(int i = 0; i < filteredRad.size(); i++){
	// 	std::cout << filteredRad[i] << " ";
	// }
	// std::cout << std::endl;
	// std::cout << filteredDetected << std::endl;
	std::cout << filteredDetected.size() << std::endl;
}
