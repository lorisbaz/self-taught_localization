#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

#include <cassert>
#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "grabcutobf.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  cv::Mat img = cv::imread("airplane.png");
  cv::Mat mask_prob = cv::imread("mask_prob.png", 0);
  cv::Mat exp_mask1 = cv::imread("exp_mask1.png", 0);
  cv::Mat exp_mask2 = cv::imread("exp_mask2.png", 0);
  
  if (img.empty() || (!mask_prob.empty() && img.size() != mask_prob.size()) ||
      (!exp_mask1.empty() && img.size() != exp_mask1.size()) ||
      (!exp_mask2.empty() && img.size() != exp_mask2.size()) ) {
    cout << "error while loading the files" << endl;
    return -1;
  }

  cv::Rect rect(cv::Point(24, 126), cv::Point(483, 294));
  cv::Mat exp_bgdModel, exp_fgdModel;
  
  cv::Mat mask;
  mask = cv::Scalar(0);
  cv::Mat bgdModel, fgdModel;
  vlg::grabCutObf( img, mask, rect, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT, -123);
  
  mask.copyTo( mask_prob );
  assert( cv::imwrite("out_mask_prob.png", mask_prob) );
  
  exp_mask1 = (mask & 1) * 255;
  assert( cv::imwrite("out_exp_mask1.png", exp_mask1) );
    
  return 0;
}
