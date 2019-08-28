#include "tum.h"
#include <memory>
#include "cuda_ros/debug_image.h"
#include "cuda_icp/icp_tracker.h"

int main(int argc, char **argv)
{
    if (argc < 2)
        return -1;

    std::string path = argv[1];
    std::vector<double> time_stamp_list;
    std::vector<std::string> depth_image_list;
    std::vector<std::string> rgb_image_list;
    std::vector<Eigen::Affine3d> ground_truth_poses;

    Eigen::Matrix3d K;
    K << 570, 0, 320,
        0, 570, 240,
        0, 0, 1;

    std::shared_ptr<ICPTracker> tracker(new ICPTracker(
        640,
        480,
        K,
        5,                   // num octave
        false,               // robust estimator
        -1,                  // debug on
        true,                // use icp
        true,                // use rgb
        100.0,               // depth cutoff
        10.0,                // max dist
        255,                 // max residual
        0,                   // min gradient
        255,                 // max gradient
        0.01,                // dist thresh
        cos(30 * 3.14 / 180) // angle thresh
        ));

    tum_load(
        path,
        time_stamp_list,
        rgb_image_list,
        depth_image_list,
        ground_truth_poses);

    cv::Mat last_depth, last_rgb;
    cv::Mat last_depth_raw, last_rgb_raw;
    cv::Mat rgb_float, rgb, depth;
    Eigen::Affine3d frame_pose;
    std::vector<Eigen::Affine3d> trajectory;

    for (int i = 0; i < time_stamp_list.size(); ++i)
    {
        cv::Mat depth_raw = cv::imread(depth_image_list[i], cv::IMREAD_UNCHANGED);
        cv::Mat rgb_raw = cv::imread(rgb_image_list[i], cv::IMREAD_UNCHANGED);

        depth_raw.convertTo(depth, CV_32FC1, 1 / 5000.f);
        rgb_raw.convertTo(rgb_float, CV_32FC3);
        cv::cvtColor(rgb_float, rgb, cv::COLOR_RGB2GRAY);

        if (!last_depth.empty() && !last_rgb.empty())
        {
            tracker->set_source_depth(cv::cuda::GpuMat(depth));
            tracker->set_source_image(cv::cuda::GpuMat(rgb));
            tracker->set_reference_depth(cv::cuda::GpuMat(last_depth));
            tracker->set_reference_image(cv::cuda::GpuMat(last_rgb));
            auto result = tracker->compute_transform(true, NULL, {10, 5, 3, 3, 3});

            cv::Mat out_cpu, out;
            cv::cuda::GpuMat out_image;
            compute_photometric_residual(
                tracker->get_vmap_ref(),
                tracker->get_image_src(),
                tracker->get_image_ref(),
                result.update,
                K,
                out_image);

            out_image.download(out_cpu);
            cv::normalize(out_cpu, out, 1.0, 0, cv::NORM_MINMAX);
            out.convertTo(out, CV_8UC1, 255);

            cv::imshow("depth", depth_raw);
            cv::imshow("rgb", rgb_raw);
            cv::imshow("pohto residual", out);

            std::cout << i << std::endl;

            compute_photometric_residual(
                tracker->get_vmap_ref(),
                tracker->get_image_src(),
                tracker->get_image_ref(),
                ground_truth_poses[i - 1].inverse() * ground_truth_poses[i],
                K,
                out_image);

            out_image.download(out_cpu);
            cv::normalize(out_cpu, out, 1.0, 0, cv::NORM_MINMAX);
            out.convertTo(out, CV_8UC1, 255);

            cv::imshow("gt residual", out);

            cv::absdiff(rgb, last_rgb, out);
            cv::normalize(out, out, 1.0, 0, cv::NORM_MINMAX);
            out.convertTo(out, CV_8UC1, 255);

            cv::imshow("residual", out);

            auto key = cv::waitKey(1);

            switch (key)
            {
            case 27:
                return 0;

            case 's':
            case 'S':
            {
            }
            break;
            }

            frame_pose = frame_pose * result.update.inverse();
        }
        else
        {
            frame_pose = ground_truth_poses[0];
        }

        trajectory.push_back(frame_pose);
        depth.copyTo(last_depth);
        rgb.copyTo(last_rgb);
        depth_raw.copyTo(last_depth_raw);
        rgb_raw.copyTo(last_rgb_raw);
    }

    save_trajectory(argv[1], time_stamp_list, trajectory);
}