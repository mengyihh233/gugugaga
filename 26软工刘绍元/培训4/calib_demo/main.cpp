#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        cv::CommandLineParser args(argc, argv,
            "{help h||}{@images|cam_mix_img|}{@output|.|}{show||}");
        if (args.has("help")) {
            std::cout << "calib_demo [image_dir] [output_dir] [--show]\n";
            return 0;
        }
        std::filesystem::path input = args.get<std::string>(0);
        std::filesystem::path output = args.get<std::string>(1);
        if (!args.check()) {
            args.printErrors();
            return 1;
        }
        if (!std::filesystem::is_directory(input)) {
            std::cerr << "Image directory not found: " << input.string() << '\n';
            return 1;
        }
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char ch) { return std::tolower(ch); });
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        const cv::Size boardSize(10, 7);
        const float squareSize = 15.0f;
        std::vector<cv::Point3f> board;
        for (int row = 0; row < boardSize.height; ++row) {
            for (int column = 0; column < boardSize.width; ++column) {
                board.emplace_back(column * squareSize, row * squareSize, 0);
            }
        }
        std::vector<std::vector<cv::Point3f>> objectPoints;
        std::vector<std::vector<cv::Point2f>> imagePoints;
        cv::Size imageSize;
        cv::Mat sample;
        std::vector<cv::Point2f> sampleCorners;
        std::string sampleName;
        std::cout << "Images: " << files.size() << std::endl;
        for (const auto& file : files) {
            cv::Mat image = cv::imread(file.string());
            if (image.empty()) {
                std::cerr << "Unreadable image: " << file.filename().string() << '\n';
                continue;
            }
            if (imageSize.empty()) imageSize = image.size();
            if (image.size() != imageSize) {
                std::cerr << "Different image size: " << file.filename().string() << '\n';
                continue;
            }
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(gray, boardSize, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found) {
                std::cout << "Skipped: " << file.filename().string() << '\n';
                continue;
            }
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001));
            imagePoints.push_back(corners);
            objectPoints.push_back(board);
            if (sample.empty()) {
                sample = image.clone();
                sampleCorners = corners;
                sampleName = file.filename().string();
            }
        }
        std::cout << "Detected: " << imagePoints.size() << " / " << files.size() << std::endl;
        if (imagePoints.size() < 6) {
            std::cerr << "At least 6 valid chessboard images are required\n";
            return 1;
        }
        cv::Mat cameraMatrix, distortion;
        std::vector<cv::Mat> rotationVectors, translationVectors;
        double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
            cameraMatrix, distortion, rotationVectors, translationVectors, cv::CALIB_USE_LU);
        if (!std::isfinite(rms) || !cv::checkRange(cameraMatrix) ||
            !cv::checkRange(distortion) || cameraMatrix.at<double>(0, 0) <= 0 ||
            cameraMatrix.at<double>(1, 1) <= 0) {
            std::cerr << "Calibration failed\n";
            return 1;
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "Image size: " << imageSize.width << " x " << imageSize.height
                  << "\nRMS: " << rms << " px\nK =\n" << cameraMatrix
                  << "\nD (k1, k2, p1, p2, k3) =\n" << distortion
                  << "\nSample: " << sampleName << std::endl;
        std::filesystem::create_directories(output);
        cv::FileStorage result((output / "calibration.yml").string(), cv::FileStorage::WRITE);
        if (!result.isOpened()) {
            std::cerr << "Cannot save calibration.yml\n";
            return 1;
        }
        result << "image_width" << imageSize.width << "image_height" << imageSize.height
               << "board_columns" << boardSize.width << "board_rows" << boardSize.height
               << "square_size_mm" << squareSize << "valid_images" << static_cast<int>(imagePoints.size())
               << "rms" << rms << "K" << cameraMatrix << "D" << distortion;
        result.release();
        cv::Mat corrected;
        cv::undistort(sample, corrected, cameraMatrix, distortion);
        cv::Mat cornersImage = sample.clone();
        cv::drawChessboardCorners(cornersImage, boardSize, sampleCorners, true);
        cv::Mat comparison(imageSize.height + 32, imageSize.width * 2, CV_8UC3, cv::Scalar(30, 30, 30));
        sample.copyTo(comparison(cv::Rect(0, 32, imageSize.width, imageSize.height)));
        corrected.copyTo(comparison(cv::Rect(imageSize.width, 32, imageSize.width, imageSize.height)));
        cv::putText(comparison, "Before", cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(comparison, "After", cv::Point(imageSize.width + 12, 23), cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if (!cv::imwrite((output / "undistort_result.jpg").string(), corrected) ||
            !cv::imwrite((output / "undistort_compare.jpg").string(), comparison) ||
            !cv::imwrite((output / "corners.jpg").string(), cornersImage)) {
            std::cerr << "Cannot save result images\n";
            return 1;
        }
        std::cout << "Saved calibration.yml, corners.jpg, undistort_result.jpg, undistort_compare.jpg"
                  << std::endl;
        if (args.has("show")) {
            cv::namedWindow("Undistortion", cv::WINDOW_NORMAL);
            cv::resizeWindow("Undistortion", 1280, 512);
            cv::moveWindow("Undistortion", 0, 0);
            cv::imshow("Undistortion", comparison);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
