# 相机标定

C++17 / OpenCV，棋盘格内角点 10×7，格长 15 mm。

```bash
cmake -S . -B build
cmake --build build -j4
./build/calib_demo /path/to/cam_mix_img . --show
```

![运行结果](shots/calibration.png)

![矫正对比](undistort_compare.jpg)
