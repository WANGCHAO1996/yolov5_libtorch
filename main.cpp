#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <time.h>

std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh = 0.15, float iou_thresh = 0.35)
{
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        //GPU推理结果为cuda数据类型，nms之前要转成cpu，否则会报错
        pred = pred.to(at::kCPU); //增加到函数里pred = pred.to(at::kCPU); 注意preds的数据类型，转成cpu进行后处理。

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({ dets.sizes()[0] });
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}


//
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <memory>
#include <fstream>
using namespace torch;
using namespace std;
using namespace cv;
int main()
{
    
    Mat img = imread("samples/000034.jpg", cv::IMREAD_GRAYSCALE);//灰度图读入
    imshow("输入", img);
    cv::waitKey(0);
    //高斯平滑滤波
    Mat imgs;
    cv::GaussianBlur(img, imgs, Size(5, 5), 0);
   /* imshow("高斯滤波", imgs);
    cv::waitKey(0);*/
    //Otsu 阈值分割
    Mat img_bw;
    cv::threshold(imgs, img_bw, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
   /* imshow("OTSU阈值分割", img_bw);
    cv::waitKey(0);*/
    //构造方形结构
    Mat SE = cv::getStructuringElement(cv::MORPH_RECT, Size(7, 7));
    //形态学开运算
    Mat img_bwop;
    cv::morphologyEx(img_bw, img_bwop, cv::MORPH_OPEN, SE, Point(-1, -1), 1);
   /* imshow("开运算", img_bwop);
    cv::waitKey(0);*/
    //用于对代表棒材区域像素灰度值累加和求阈值，以便确定棒材区域起止行、列下标
    int thr1 = 5;
    //# 垂直投影，计算二值图像每列的灰度值之和,注意数据类型，避免溢出
    //	计算垂直投影
    int* colheight = new int[img_bwop.cols];
    //数组必须赋初值为零，否则出错。无法遍历数组。 
    vector <int> array;//动态数组用来存储投影值大于阈值的横坐标
    memset(colheight, 0, img_bwop.cols * 4);
    int value;
    for (int i = 0; i < img_bwop.rows; i++)
    {
        for (int j = 0; j < img_bwop.cols; j++)
        {
            value = img_bwop.at<uchar>(i, j);
            if (value == 255)
            {
                colheight[j]++;
            }

        }
    }

    Mat histogramImage(img_bwop.rows, img_bwop.cols, CV_8UC1);
    for (int i = 0; i < img_bwop.rows; i++)
    {
        for (int j = 0; j < img_bwop.cols; j++)
        {
            value = 0;  //设置为黑色。  
            histogramImage.at<uchar>(i, j) = value;
        }
    }

    for (int i = 0; i < img_bwop.cols; i++)
    {
        for (int j = 0; j < colheight[i]; j++)
        {
            value = 255;  //设置为白色  
            histogramImage.at<uchar>(j, i) = value;
        }
    }

   /* imshow("垂直投影", histogramImage);
    waitKey(0);*/

    Mat lineImage(img_bwop.rows, img_bwop.cols, CV_8UC1, cv::Scalar(0, 0, 0));

    //寻找投影大于阈值5的横坐标
    for (int i = 0; i < img_bwop.cols; i++)
    {
        bool flag = true;

        for (int j = 0; j < colheight[i] && colheight[i] >= 5; j++)
        {

            if (flag == true)
            {
                array.push_back(i);
                flag = false;
            }
        }
    }
    int count = array.size();
    cout << count << endl;
    int min = *min_element(array.begin(), array.end());//最小列
    int max = *max_element(array.begin(), array.end());//最大列
    cout << min << endl;
    cout << max << endl;
    //计算水平投影
    int* colheighttwo = new int[img_bwop.rows];
    //数组必须赋初值为零，否则出错。无法遍历数组。 
    vector <int> array1;
    memset(colheighttwo, 0, img_bwop.rows * 4);
    int valuetwo;
    for (int i = 0; i < img_bwop.rows; i++)
        for (int j = 0; j < img_bwop.cols; j++)
        {
            valuetwo = img_bwop.at<uchar>(i, j);
            if (valuetwo == 255)
            {
                colheighttwo[i]++;
            }
        }

    Mat plantImage(img_bwop.rows, img_bwop.cols, CV_8UC1);  //创建一个新的mat型
                                                            //把这个图全部画成黑色
    for (int i = 0; i < img_bwop.rows; i++)
    {
        for (int j = 0; j < img_bwop.cols; j++)
        {
            valuetwo = 0;  //设置为黑色。  
            plantImage.at<uchar>(i, j) = valuetwo;
        }
    }

    for (int i = 0; i < img_bwop.rows; i++)
    {
        for (int j = 0; j < colheighttwo[i]; j++)
        {
            valuetwo = 255;  //设置为白色  
            plantImage.at<uchar>(i, j) = valuetwo;
        }
    }
    /*imshow("水平投影", plantImage);
    waitKey(0);*/
    Mat lineImage1(plantImage.rows, plantImage.cols, CV_8UC1, cv::Scalar(0, 0, 0));

    //寻找投影大于阈值5的纵坐标
    for (int i = 0; i < plantImage.rows; i++)
    {
        bool flag = true;
        for (int j = 0; j < colheighttwo[i] && colheighttwo[i] >= 5; j++)
        {
            if (flag == true)
            {
                array1.push_back(i);
                flag = false;
            }

        }
    }
    int count1 = array1.size();
    cout << count1 << endl;
    int min1 = *min_element(array1.begin(), array1.end());//最小行
    int max1 = *max_element(array1.begin(), array1.end());//最大行
    cout << min1 << endl;
    cout << max1 << endl;
    delete[] colheighttwo;//释放前面申请的空
    cv::Mat image_part = img(cv::Rect(min, min1, max - min, max1 - min1)); // 裁剪后的图
    imshow("剪切后不扩张：", image_part);
    waitKey(0);
    cv::Mat image_part1 = img(cv::Rect(MAX(min - 25, 0), MAX(min1 - 25, 0), MIN(max - min + 50, img_bwop.cols), MIN(max1 - min1 + 50, img_bwop.rows))); // 裁剪后的图扩张
    /*imshow("剪切后扩张：", image_part1);
    waitKey(0);*/

    //加载模型
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    torch::DeviceType device_type = at::kCPU; // 定义设备类型
    if (torch::cuda::is_available())
        device_type = at::kCUDA;
    // Loading  Module
    torch::jit::script::Module module = torch::jit::load("best.torchscript3.pt");//yolov5x.torchscript.pt//best.torchscript.pt
    module.to(device_type); // 模型加载至GPU

    std::vector<std::string> classnames;
    std::ifstream f("obj.names");//自定义names文件
    std::string name = "";
    while (std::getline(f, name))
    {
        classnames.push_back(name);
    }

    ofstream outfile;
    outfile.open("data.txt");//创建txt文件

    cv::Mat outimg;
   //Mat image_part = imread("000000.jpg");
  
    // Preparing input tensor
    cv::resize(image_part, outimg, cv::Size(640, 352));
   /* imshow("outimg:", outimg);
    waitKey(0);*/
    cv::cvtColor(outimg, outimg, cv::COLOR_BGR2RGB);  // BGR -> RGB
    outimg.convertTo(outimg, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto imgTensor = torch::from_blob(outimg.data, { 1, outimg.rows, outimg.cols, outimg.channels() }).to(device_type);
    imgTensor = imgTensor.permute({ 0, 3, 1, 2 }).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(imgTensor);
    // preds: [?, 15120, 9]
    torch::jit::IValue output = module.forward(inputs);
    auto preds = output.toTuple()->elements()[0].toTensor();
    // torch::Tensor preds = module.forward({ imgTensor }).toTensor();
    std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.1, 0.2);
    if (dets.size() > 0)
    {
        // Visualize result
        int total = 0;
        for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
        {
            
            float left = dets[0][i][0].item().toFloat() * image_part.cols / 640;
            float top = dets[0][i][1].item().toFloat() * image_part.rows / 352;
            float right = dets[0][i][2].item().toFloat() * image_part.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * image_part.rows / 352;
            float score = dets[0][i][4].item().toFloat();
            int classID = dets[0][i][5].item().toInt();

            cv::rectangle(image_part, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);//画框
            cv::circle(image_part, Point(int((left + right) / 2), int((top + bottom) / 2)), 2, cv::Scalar(0, 0, 0), 2, 8, 0);//画圆心

            outfile << Point(int((left + right) / 2)+min, int((top + bottom) / 2)+min1) << "\n";//坐标写入txt

            cv::putText(image_part,
                classnames[classID] + ": " + cv::format("%.2f", score),
                cv::Point(left, top),
                cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            total = total + 1;
        }
        outfile.close();//关闭txt文件，保存
        String count = "count:" + to_string(total);
        cv::putText(image_part,count,Point(28,38), cv::FONT_HERSHEY_SIMPLEX,1.5, cv::Scalar(255, 0, 255),2,8,0);//打印检测到个数
        cv::imshow("out", image_part);
        cv::imwrite("out/detect.jpg", image_part);
        waitKey(0);
    }
    
    return 0;
        
}
