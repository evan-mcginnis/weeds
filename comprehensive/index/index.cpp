// index.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "stdafx.h"
#include <iostream>

#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace cv;
using namespace std;

// Driver code 
int main(int argc, char** argv)
{
    // Read the image file as 
    // imread("default.jpg"); 
    Mat image = imread("Enter the Address"
        "of Input Image",
        IMREAD_GRAYSCALE);

    // Error Handling 
    if (image.empty()) {
        cout << "Image File "
            << "Not Found" << endl;

        // wait for any key press 
        cin.get();
        return -1;
    }

    // Show Image inside a window with 
    // the name provided 
    imshow("Window Name", image);

    // Wait for any keystroke 
    waitKey(0);
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
