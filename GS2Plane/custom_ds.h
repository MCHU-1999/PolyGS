#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Plane_3 Plane_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

// Define a tuple to hold all Gaussian Splat properties
typedef std::tuple<Point_3, Vector_3, double, double, double, 
                   double, double, double, 
                   double, double, double, double> GaussianTuple;

struct GaussianSplat {
  double x, y, z;
  double nx, ny, nz;
  double f_dc_0, f_dc_1, f_dc_2;
  double scale_0, scale_1, scale_2;
  double rot_0, rot_1, rot_2, rot_3;
};

struct Rectangle3D {
  Point_3 center;
  Point_3 v1, v2, v3, v4;
  Vector_3 normal;
  Vector_3 axis1, axis2;
  double pseudo_radius;
  double width, height;
  double red, green, blue;
};