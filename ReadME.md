使用libtorch部署yolov5模型技术说明

环境说明：vs2019+opencv4.5+libtorch1.7.1
1.VS2019下载网址：https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/
2.opencv下载：http://opencv.org/下载最新版本即可。
3.Libtorch下载：https://pytorch.org/ 选择适合自己环境的cuda版本，推荐下载release版本。

环境搭建参考：https://blog.csdn.net/weixin_44936889/article/details/111186818
https://blog.csdn.net/zzz_zzz12138/article/details/109138805
https://blog.csdn.net/wenghd22/article/details/112512231
vs2019和opencv配置过程参考链接；
https://blog.csdn.net/sophies671207/article/details/89854368
给出我的libtorch配置过程：新建空项目 新建main.cpp文件
1新建项目->属性->VC++目录->包含目录

2新建项目->属性->VC++目录->库目录


3新建项目->属性->C/C++目录->常规->附加包含目录

4新建项目->属性->C/C++目录->常规->SDL检查 ：否

5新建项目->属性->连接器->输入->附加依赖项：写入以下
E:\opencv\build\x64\vc15\lib\opencv_world450.lib
c10.lib
asmjit.lib
c10_cuda.lib
caffe2_detectron_ops_gpu.lib
caffe2_module_test_dynamic.lib
caffe2_nvrtc.lib
clog.lib
cpuinfo.lib
dnnl.lib
fbgemm.lib
libprotobuf.lib
libprotobuf-lite.lib
libprotoc.lib
mkldnn.lib
torch.lib
torch_cuda.lib
torch_cpu.lib
kernel32.lib
user32.lib
gdi32.lib
winspool.lib
comdlg32.lib
advapi32.lib
shell32.lib
ole32.lib
oleaut32.lib
uuid.lib
odbc32.lib
odbccp32.lib

6新建项目->属性->连接器->命令行：输入/INCLUDE:?warp_size@cuda@at@@YAHXZ

7新建项目->属性->C/C++目录->语言->符合模式 ：否

运行前将libtorch lib中的全部文件复制进入G:\newdetect\x64\Release下。

配置好了以上环境，打包好的文件夹如下图：

权重文件：best.torchscript3.pt  直接运行main.cpp即可。采用的samples文件夹下的图片进行测试。

检测图片保存在out文件夹下，坐标保存在data.txt中。

