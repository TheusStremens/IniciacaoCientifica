#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;
const string PATH_CASCADE_FACE_IR = "/home/matheusm/Cascades/cascade.xml";
const char *windowName = "Demo";
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnInit;

bool protonect_shutdown = false;

libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
libfreenect2::Registration* registration;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
void onMouse(int event, int x, int y, int, void*)
{
  // We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
  if (event != CV_EVENT_LBUTTONDOWN)
    return;
  // Check if the user clicked on one of our GUI buttons.
  Point pt = Point(x,y);
  if (isPointInRect(pt, m_rcBtnAdd)) {
      cout << "User clicked [Add Person] button when numPersons was "  << endl;
  }
  else if (isPointInRect(pt, m_rcBtnDel)) {
      cout << "User clicked [Delete All] button." << endl;
  }
  else if (isPointInRect(pt, m_rcBtnInit)) {
      cout << "User clicked [Debug] button." << endl;
  }
}

// Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
// and/or negative y coords for bottom-justified text.
// Returns the bounding rect around the drawn text.
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.45f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Get the text size & baseline.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Adjust the coords for left/right-justified or top/bottom-justified.
    if (coord.y >= 0) {
        // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
        coord.y += textSize.height;
    }
    else {
        // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
        coord.y += img.rows - baseline + 1;
    }
    // Become right-justified if desired.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    // Get the bounding box around the text.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // Let the user know how big their text is, in case they want to arrange things.
    return boundingRect;
}

// Draw a GUI button into the image, using drawString().
// Can specify a minWidth if you want several buttons to all have the same width.
// Returns the bounding rect around the drawn button, allowing you to position buttons next to each other.
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

void TrainRecognizeAuthenticateUsingKinect(libfreenect2::SyncMultiFrameListener &listener, libfreenect2::FrameMap &frames) {
  protonect_shutdown = false;
  //loop kinect  
  while(!protonect_shutdown)
  {
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *color = frames[libfreenect2::Frame::Color];
    Mat depth_image = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f;
    Mat ir_image = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data) / 80000.0f;
    Mat color_image = cv::Mat(color->height, color->width, CV_8UC4, color->data);
    cv::resize(color_image, color_image, Size(color_image.cols/2, color_image.rows/2), 1.0, 1.0, INTER_CUBIC);

    registration->apply(color, depth, &undistorted, &registered);
    Mat registered_image = cv::Mat(registered.height, registered.width, CV_8UC4, registered.data);
    
    // Get a copy of the camera frame that we can draw onto.
    Mat displayedFrame;
    registered_image.copyTo(displayedFrame);
    vector< Rect_<int> > faces_ir;
    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE_IR);
    ir_image.convertTo(ir_image, CV_8UC3, 255, 0);
    haar_cascade.detectMultiScale(ir_image, faces_ir);
    for(int i = 0; i < faces_ir.size(); i++) {
      Rect face_i = faces_ir[i];
      rectangle(displayedFrame, face_i, CV_RGB(0, 255, 20), 2);
    }

    // Draw the GUI buttons into the main image.
    m_rcBtnAdd = drawButton(displayedFrame, "Cadastrar Pessoa", Point(BORDER, BORDER+30));
    m_rcBtnDel = drawButton(displayedFrame, "Deletar Todos", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height + 1), m_rcBtnAdd.width);
    m_rcBtnInit = drawButton(displayedFrame, "Iniciar Sistema", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height+1), m_rcBtnAdd.width);
    // Create a GUI window for display on the screen.
    namedWindow(windowName); // Resizable window, might not work on Windows.
    // Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
    setMouseCallback(windowName, onMouse, 0);
    
    imshow(windowName, displayedFrame);

    int key = cv::waitKey(1);
    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
    listener.release(frames);
  }

}

int main(int argc, char *argv[])
{
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("Protonect");
  std::string binpath = "/";
  if(executable_name_idx != std::string::npos)
    binpath = program_path.substr(0, executable_name_idx);
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;
  if(freenect2.enumerateDevices() == 0)
  {
    std::cout << "no device connected!" << std::endl;
    return -1;
  }
  std::string serial = freenect2.getDefaultDeviceSerialNumber();
  bool viewer_enabled = true;
  for(int argI = 1; argI < argc; ++argI)
  {
    const std::string arg(argv[argI]);
    if(arg == "cpu")
    {
      if(!pipeline)
        pipeline = new libfreenect2::CpuPacketPipeline();
    }
    else if(arg == "gl")
    {
  #ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
  #else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
  #endif
    }
    else if(arg == "cl")
    {
  #ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline();
  #else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
  #endif
    }
    else if(arg.find_first_not_of("0123456789") == std::string::npos) //check if parameter could be a serial number
    {
      serial = arg;
    }
    else if(arg == "-noviewer")
    {
      viewer_enabled = false;
    }
    else
    {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
  }
  if(pipeline)
  {
    dev = freenect2.openDevice(serial, pipeline);
  }
  else
  {
    dev = freenect2.openDevice(serial);
  }
  if(dev == 0)
  {
    std::cout << "failure opening device!" << std::endl;
    return -1;
  }
  signal(SIGINT,sigint_handler);
  
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;
  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  dev->start();
  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
  registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());

  TrainRecognizeAuthenticateUsingKinect(listener, frames);
  
  dev->stop();
  dev->close();
  return 0;
}
