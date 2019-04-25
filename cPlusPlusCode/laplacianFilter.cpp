#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(){

    //Read image
    Mat image = imread("../assets/crop1.png", IMREAD_GRAYSCALE);

    //check is image exits
    if(image.empty()){
        cout<<"can not find image"<<endl;
        return 0;
    }

    Mat laplacian, logimg;

    //Apply laplacian
    Laplacian(image, laplacian, CV_32F, 3, 1, 0);

    Mat LOGKernel = (Mat_<double>(3,3) <<  0.4038, 0.8021, 0.4038, 0.8021, -4.8233, 0.8021, 0.4038, 0.8021, 0.4038);

    //filter image using kernel
    filter2D(image, logimg, CV_32F, LOGKernel);

    //normalize images to display
    normalize(laplacian, laplacian, 0, 1, NORM_MINMAX);
    normalize(logimg, logimg, 0, 1, NORM_MINMAX);

    //create windows to display images
    namedWindow("image", WINDOW_NORMAL);
    namedWindow("laplacian", WINDOW_NORMAL);
    namedWindow("log", WINDOW_NORMAL);

    //show images
    imshow("image", image);
    imshow("laplacian", laplacian);
    imshow("log", logimg);
    
    //press esc to exit the program
    waitKey(0);

    //close all the opened windows
    destroyAllWindows();
    
    return 0;
}