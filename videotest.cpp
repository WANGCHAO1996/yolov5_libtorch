#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <time.h>

std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh = 0.01, float iou_thresh = 0.35)
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

#include <torch/script.h> 
#include <iostream>
#include <memory>
//int main(int argc, const char* argv[]) {
//    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
//    torch::DeviceType device_type = at::kCPU; // 定义设备类型
//    if (torch::cuda::is_available())
//        device_type = at::kCUDA;
//}


int main(int argc, char* argv[])
{
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    torch::DeviceType device_type = at::kCPU; // 定义设备类型
    if (torch::cuda::is_available())
        device_type = at::kCUDA;
    // Loading  Module
    torch::jit::script::Module module = torch::jit::load("best.torchscript3.pt");//best.torchscript3.pt//yolov5x.torchscript.pt
    module.to(device_type); // 模型加载至GPU

    std::vector<std::string> classnames;
    std::ifstream f("obj.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        classnames.push_back(name);
    }
    if (argc < 2)
    {
        std::cout << "Please run with test video." << std::endl;
        return -1;
    }
    std::string video = argv[1];
    cv::VideoCapture cap = cv::VideoCapture(video);
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat frame, img;
    cap.read(frame);
    int width = frame.size().width;
    int height = frame.size().height;
    int count = 0;
    while (cap.isOpened())
    {
        count++;
        clock_t start = clock();
        cap.read(frame);
        if (frame.empty())
        {
            std::cout << "Read frame failed!" << std::endl;
            break;
        }

        // Preparing input tensor
        cv::resize(frame, img, cv::Size(640, 352));
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
        // imgTensor = imgTensor.permute({2,0,1});
        // imgTensor = imgTensor.toType(torch::kFloat);
        // imgTensor = imgTensor.div(255);
        // imgTensor = imgTensor.unsqueeze(0);
        // imgTensor = imgTensor.to(device_type);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // BGR -> RGB
        img.convertTo(img, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
        auto imgTensor = torch::from_blob(img.data, { 1, img.rows, img.cols, img.channels() }).to(device_type);
        imgTensor = imgTensor.permute({ 0, 3, 1, 2 }).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(imgTensor);
        // preds: [?, 15120, 9]
        torch::jit::IValue output = module.forward(inputs);
        auto preds = output.toTuple()->elements()[0].toTensor();
        // torch::Tensor preds = module.forward({ imgTensor }).toTensor();
        std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.35, 0.5);
        if (dets.size() > 0)
        {
            // Visualize result
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
            {
                float left = dets[0][i][0].item().toFloat() * frame.cols / 640;
                float top = dets[0][i][1].item().toFloat() * frame.rows / 352;
                float right = dets[0][i][2].item().toFloat() * frame.cols / 640;
                float bottom = dets[0][i][3].item().toFloat() * frame.rows / 352;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

                cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

                cv::putText(frame,
                    classnames[classID] + ": " + cv::format("%.2f", score),
                    cv::Point(left, top),
                    cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            }
        }
        // std::cout << "-[INFO] Frame:" <<  std::to_string(count) << " FPS: " + std::to_string(float(1e7 / (clock() - start))) << std::endl;
        std::cout << "-[INFO] Frame:" << std::to_string(count) << std::endl;
        // cv::putText(frame, "FPS: " + std::to_string(int(1e7 / (clock() - start))),
        //     cv::Point(50, 50),
        //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("", frame);
        // cv::imwrite("../images/"+cv::format("%06d", count)+".jpg", frame);
        cv::resize(frame, frame, cv::Size(width, height));
        if (cv::waitKey(1) == 27) break;
    }
    cap.release();
    return 0;
}
