#ifndef _GRABCUTOBF_HPP_
#define _GRABCUTOBF_HPP_

#include <opencv2/opencv.hpp>

namespace vlg {

  void grabCutObf( cv::InputArray img, cv::InputOutputArray mask, cv::Rect rect,
		   cv::InputOutputArray bgdModel, cv::InputOutputArray fgdModel,
		   int iterCount, int mode, int TEST);

} // end namespace

#endif  // _GRABCUTOBF_HPP_

