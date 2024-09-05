#include <Eigen/Dense>
#include <iostream>

int main() {
  // Create a 3x3 matrix and fill it with values
  Eigen::Matrix3d m;
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Eigen::VectorXi test(30);

  test(29) = 20;

  // Print the matrix
  std::cout << "Matrix m:" << std::endl << m << std::endl;

  // Create a vector and fill it with values
  Eigen::Vector3d v(1, 2, 3);
  v = v.transpose();

  // Print the vector
  std::cout << "Vector v:" << std::endl << v << std::endl;

  // Multiply the matrix and vector
  Eigen::Vector3d result = m * v;

  // Print the result
  std::cout << "Result of m * v:" << std::endl << result << std::endl;

  return 0;
}
