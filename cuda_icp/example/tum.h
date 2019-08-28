#ifndef TUM_LOADER_H
#define TUM_LOADER_H

#include <vector>
#include <string>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

int find_closest_index(const std::vector<double> &list, double time)
{
    int idx = -1;
    double min_val = std::numeric_limits<double>::max();
    for (int i = 0; i < list.size(); ++i)
    {
        double d = std::abs(list[i] - time);
        if (d < min_val)
        {
            idx = i;
            min_val = d;
        }
    }

    return idx;
}

void tum_load(
    const std::string &path,
    std::vector<double> &time_stamp_list,
    std::vector<std::string> &depth_image_list,
    std::vector<std::string> &rgb_image_list,
    std::vector<Eigen::Affine3d> &ground_truth_poses)
{
    std::ifstream file(path + "association.txt");

    if (file.is_open())
    {
        double ts;
        std::string depth_image;
        std::string rgb_image;

        while (file >> ts >> depth_image >> ts >> rgb_image)
        {
            depth_image_list.push_back(path + depth_image);
            rgb_image_list.push_back(path + rgb_image);
            time_stamp_list.push_back(ts);
        }
    }

    file.close();
    file.open(path + "groundtruth.txt");

    if (file.is_open())
    {
        double ts;
        double tx, ty, tz, qx, qy, qz, qw;
        std::vector<double> all_gt_time_stamp;
        std::vector<Eigen::Affine3d> all_gt_readings;

        for (int i = 0; i < 3; ++i)
        {
            std::string line;
            std::getline(file, line);
            std::cout << line << std::endl;
        }

        while (file >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            Eigen::Quaterniond rotation(qw, qx, qy, qz);
            auto translation = Eigen::Vector3d(tx, ty, tz);

            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.topLeftCorner(3, 3) = rotation.toRotationMatrix();
            T.topRightCorner(3, 1) = translation;

            Eigen::Affine3d gt(T);
            all_gt_time_stamp.push_back(ts);
            all_gt_readings.push_back(gt);
        }

        int last_idx = 0;

        for (int i = 0; i < time_stamp_list.size(); ++i)
        {
            double time = time_stamp_list[i];
            int idx = find_closest_index(all_gt_time_stamp, time);

            if (idx <= last_idx)
                std::cout << "ERROR: index " << idx << " last_idx: " << last_idx << std::endl;

            ground_truth_poses.push_back(all_gt_readings[idx]);
            last_idx = idx;
        }
    }

    std::cout << "total of " << time_stamp_list.size() << " images loaded." << std::endl;
}

void save_trajectory(
    const std::string &path,
    std::vector<double> time_stamp_list,
    std::vector<Eigen::Affine3d> trajectory)
{
    std::ofstream file(path + "result.txt", std::ios_base::out);

    if (file.is_open())
    {
        for (int i = 0; i < trajectory.size(); ++i)
        {
            if (i >= time_stamp_list.size())
                break;

            double ts = time_stamp_list[i];
            const auto &pose = trajectory[i];
            Eigen::Vector3d t = pose.translation();
            Eigen::Quaterniond q(pose.rotation());

            file << std::fixed
                 << std::setprecision(4)
                 << ts << " "
                 << t(0) << " "
                 << t(1) << " "
                 << t(2) << " "
                 << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << std::endl;
        }

        file.close();
    }
}

#endif