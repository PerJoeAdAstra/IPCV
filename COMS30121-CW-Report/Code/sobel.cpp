#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void convolution(
	cv::Mat &input,
	cv::Mat &kernel,
	cv::Mat &output);

void sobel(
  cv::Mat &input,
  cv::Mat &xDerivative,
  cv::Mat &yDerivative,
  cv::Mat &gradMag,
  cv::Mat &gradDir);

int main(int argc, char** argv)
{
  // LOADING THE IMAGE
  char* imageName = argv[1];

  Mat image;
  image = imread( imageName, 1 );

  if( argc != 2 || !image.data )
  {
    printf( " No image data \n " );
    return -1;
  }

  // CONVERT COLOUR, BLUR AND SAVE
  Mat gray_image;
  cvtColor( image, gray_image, CV_BGR2GRAY );

  Mat xDerivative;
  Mat yDerivative;
  Mat gradMag;
  Mat gradDir;
  sobel(gray_image, xDerivative, yDerivative, gradMag, gradDir);

  xDerivative.convertTo(xDerivative, CV_32F);
  yDerivative.convertTo(yDerivative, CV_32F);
  magnitude(xDerivative, yDerivative, gradMag);

  imwrite("xDerivative.jpg", xDerivative);
  imwrite("yDerivative.jpg", yDerivative);
  imwrite("gradMag.jpg", gradMag);
  imwrite("gradDir.jpg", gradDir);

  return 0;
}

void sobel(cv::Mat &input, cv::Mat &xDerivative, cv::Mat &yDerivative,
           cv::Mat &gradMag,cv::Mat &gradDir)
{
  Mat_<int> xKernel(3, 3, CV_8SC1);
  xKernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;

  Mat_<int> yKernel(3, 3, CV_8SC1);
  yKernel << -1, -2, -1, 0, 0, 0, 1, 2, 1;

  convolution(input, xKernel, xDerivative);
  convolution(input, yKernel, yDerivative);
}

void convolution(cv::Mat &input, cv::Mat &kernel, cv::Mat &output)
{

  output.create(input.size(), input.type());

  // we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using

					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = -m + kernelRadiusX;
					int kernely = -n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<int>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
      if(sum > 255) sum =  255;
      if(sum < 0) sum = 0;
			output.at<uchar>(i, j) = (double) sum;
    }
  }
}
