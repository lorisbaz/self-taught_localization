#include <Python.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <numpy/ndarrayobject.h>

#include "grabcutobf.hpp"
#include "cv2_common.hpp"

using namespace cv;

static PyObject* pyopencv_grabCutObf(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_rect = NULL;
    Rect rect;
    PyObject* pyobj_bgdModel = NULL;
    Mat bgdModel;
    PyObject* pyobj_fgdModel = NULL;
    Mat fgdModel;
    int iterCount=0;
    int mode = GC_EVAL;
    const char* keywords[] = { "img", "mask", "rect", "bgdModel", \
                               "fgdModel", "iterCount", "mode", NULL };

//         printf("1\n"); fflush(stdout);
// PyArg_ParseTuple(args, "OOOOOii", \
//                    &pyobj_img, &pyobj_mask, \
//                    &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, \
// 			    &iterCount, &mode);

//  pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0));
//         printf("5\n"); fflush(stdout);

// 	if( !PyArray_Check(pyobj_mask) ) {
// 	  printf("BAAAAAAD\n"); fflush(stdout);
// 	}
//  pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1));
//         printf("4\n"); fflush(stdout);

// // PyArg_ParseTupleAndKeywords(args, kw, "OOOOOii:grabCutObf", \
// //                    (char**)keywords, &pyobj_img, &pyobj_mask, \
// //                    &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, \
// // 			    &iterCount, &mode);
//         printf("2\n"); fflush(stdout);
//  pyopencv_to(pyobj_img, img, ArgInfo("img", 0));
//         printf("3\n"); fflush(stdout);
//  pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1));
//         printf("4\n"); fflush(stdout);
//  pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0));
//         printf("5\n"); fflush(stdout);
//  pyopencv_to(pyobj_bgdModel, bgdModel, ArgInfo("bgdModel", 1));
//         printf("6\n"); fflush(stdout);
//  pyopencv_to(pyobj_fgdModel, fgdModel, ArgInfo("fgdModel", 1));
//         printf("7\n"); fflush(stdout);

    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|i:grabCutObf", \
                   (char**)keywords, &pyobj_img, &pyobj_mask, \
                   &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, \
                   &iterCount, &mode)  &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) &&
        pyopencv_to(pyobj_bgdModel, bgdModel, ArgInfo("bgdModel", 1)) &&
        pyopencv_to(pyobj_fgdModel, fgdModel, ArgInfo("fgdModel", 1)) )
    {
      ERRWRAP2( vlg::grabCutObf(img, mask, rect, bgdModel,		\
                               fgdModel, iterCount, mode, -123));
      Py_RETURN_NONE;
    }

    return NULL;
}

static PyMethodDef GrabCutObfMethods[] = {
  { "grabCutObf", \
    (PyCFunction)pyopencv_grabCutObf, \
    METH_KEYWORDS, \
    "grabCutObf(img, mask, rect, bgdModel, fgdModel, "\
               "iterCount, mode, TEST) -> None"},
  { NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
initgrabcutobf(void)
{
    (void) Py_InitModule("grabcutobf", GrabCutObfMethods);
    import_array();
}
