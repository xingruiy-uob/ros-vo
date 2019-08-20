#ifndef EXP_MAP_H
#define EXP_MAP_H

#include <Eigen/Geometry>

template <typename Derived>
Eigen::Matrix3d rodrigues_decompress(const Eigen::MatrixBase<Derived> &xi)
{
  double a = xi(0);
  double b = xi(1);
  double c = xi(2);
  double theta = xi.norm();

  if (std::isnan(theta) || theta < std::numeric_limits<double>::epsilon())
    return Eigen::Matrix3d::Identity();

  Eigen::Matrix3d A, B;

  /*
  $$ A = \hat{\xi} $$
  */
  A << 0, -c, b,
      c, 0, -a,
      -b, a, 0;

  /*
  $$ B = A^2 $$
  */
  B << -c * c - b * b, a * b, a * c,
      a * b, -c * c - a * a, b * c,
      a * c, b * c, -b * b - a * a;

  /*
  $$ R = I_3 + \frac{sin(\theta)}{\theta}A + \frac{1-cos(\theta)}{\theta^2}B $$
  */
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin(theta) / theta * A + (1 - cos(theta)) / pow(theta, 2) * B;

  return R;
}

template <typename Derived>
Eigen::Affine3d se3_exp_map(const Eigen::MatrixBase<Derived> &xi)
{
  const auto R = rodrigues_decompress(xi.bottomRows(3));
  const Eigen::Vector3d t = xi.topRows(3);
  auto se3 = Eigen::Affine3d::Identity();
  se3.rotate(R);
  se3.translate(t);
  return se3;
}

#endif