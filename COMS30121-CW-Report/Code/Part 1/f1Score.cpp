/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
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
#include <fstream>
#include <tuple>

using namespace std;
using namespace cv;

/** Function Headers */
void calcF1Score( Mat frame, std::vector<tuple<tuple<int, int>, int, int>> groundTruth);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  // 2. Read input groundtruth text file
  String filename = argv[2];
  ifstream groundTruthFile;
  groundTruthFile.open(filename);

  if (!groundTruthFile)
  {
    cerr << "Unable to open file " + filename;
    exit(1);
  }

  // Store a vector of tuples for the parsed ground truth
  std::vector<tuple <tuple<int, int>, int, int>> groundTruth;
  String line;

  while(getline(groundTruthFile, line))
  {
    int x; // The x coordinate of the upper left corner of the ground truth bounding box
    int y; // The y coordinate of the upper left corner of the ground truth bounding box
    int width; // The width of the ground truth bounding box
    int height; // The height of the ground truth bounding box
    int elementCount = 0; // The number of ground truth elements parsed from a line in the ground truth file
    char c; // A place to store each character as it is read from the ground truth file
    int tempStore = 0; // A temporary store for multi-digit numbers in the ground truth file
                       // as they are read one character at a time

    for(int i = 0; i < line.length(); i++)
    {
      c = line.at(i); // Read in character
      // If character is a number, add it to temp store
      if(c != '(' && c != ')' && c != ',')
      {
        // Numbers in ground truth file are decimal, so needs to be shifted by
        // factor of 10 every time a new digit is read in.
        // c - '0' is to get the int value of the number character
        tempStore = tempStore * 10 + (c - '0');
      }
      // If character is a comma or closing parenthesis, store the previously parsed
      // value in the corresponding variable
      else if((c == ',' || c == ')') && tempStore != 0)
      {
        // Switch on how many elements have already been parsed
        switch(elementCount)
        {
          case 0:
            x = tempStore;
            tempStore = 0;
            break;
          case 1:
            y = tempStore;
            tempStore = 0;
            break;
          case 2:
            width = tempStore;
            tempStore = 0;
            break;
          case 3:
            height = tempStore;
            tempStore = 0;
            break;
          default:
            break;
        }
        // Increment the number of elements parsed
        elementCount++;
      }
    }
    // Add the ground truth tuple to the vector of ground truths
    groundTruth.push_back(make_tuple(make_tuple(x,y), width, height));
  }
  groundTruthFile.close();

	// 3. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 4. Detect Faces and Display Result
	calcF1Score( frame, groundTruth );

	return 0;
}

/** @function detectAndDisplay */
void calcF1Score( Mat frame, std::vector<tuple<tuple<int, int>, int, int>> groundTruth )
{
	std::vector<Rect> faces;
	Mat frame_gray;
  int threshold = 30; // Threshold to allow bounding boxes to differ by
  double f1Score = -1;
  double tpCount = 0; // Number of True Positives
  double fpCount = 0; // Number of False Positives
  double fnCount = 0; // Number of False Negatives

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
    // Compare with ground truth
    for( int j = 0; j < groundTruth.size(); j++)
    {
      // Get the difference between each of the four points of the two rectangles
      int x1Diff = faces[i].x - get<0>(get<0>(groundTruth[j]));
      int y1Diff = faces[i].y - get<1>(get<0>(groundTruth[j]));
      int x2Diff = (faces[i].x + faces[i].width) -
                          (get<0>(get<0>(groundTruth[j])) + get<1>(groundTruth[j]));
      int y2Diff = (faces[i].y + faces[i].height) -
                          (get<1>(get<0>(groundTruth[j])) + get<2>(groundTruth[j]));
      // If all of the differences are below the threshold, it is a True Positive
      if( abs(x1Diff) <= threshold && abs(y1Diff) <= threshold &&
            abs(x2Diff) <= threshold && abs(y2Diff) <= threshold)
      {
        tpCount++;
      }
    }
	}

  // Any faces identified in the image that aren't True Positives are instead
  // False Positives
  fpCount = faces.size() - tpCount;
  // Any faces that aren't identified in the image are False Negatives
  fnCount = groundTruth.size() - tpCount;

  // 4. Calculate the F1 score using a simplified equation
  // F1 = (recall * precision) / (recall + precision) can be simplified to the
  // following
  if(tpCount != 0 || fpCount != 0 || fnCount != 0)
  {
    f1Score = (2 * tpCount) / (2 * tpCount + fpCount + fnCount);
  }

  // 5. Print the F1 score with 4 decimal places
  std::cout.precision(4);
  std::cout << f1Score << std::endl;
}
