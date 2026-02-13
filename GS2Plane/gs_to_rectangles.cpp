#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

// Define a tuple to hold all Gaussian Splat properties
typedef std::tuple<Point_3, Vector_3, double, double, double, 
                   double, double, double, 
                   double, double, double, double> GaussianTuple;

// Property maps for accessing tuple elements
typedef CGAL::Nth_of_tuple_property_map<0, GaussianTuple> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, GaussianTuple> Normal_map;
typedef CGAL::Nth_of_tuple_property_map<2, GaussianTuple> F_dc_0_map;
typedef CGAL::Nth_of_tuple_property_map<3, GaussianTuple> F_dc_1_map;
typedef CGAL::Nth_of_tuple_property_map<4, GaussianTuple> F_dc_2_map;
typedef CGAL::Nth_of_tuple_property_map<5, GaussianTuple> Scale_0_map;
typedef CGAL::Nth_of_tuple_property_map<6, GaussianTuple> Scale_1_map;
typedef CGAL::Nth_of_tuple_property_map<7, GaussianTuple> Scale_2_map;
typedef CGAL::Nth_of_tuple_property_map<8, GaussianTuple> Rot_0_map;
typedef CGAL::Nth_of_tuple_property_map<9, GaussianTuple> Rot_1_map;
typedef CGAL::Nth_of_tuple_property_map<10, GaussianTuple> Rot_2_map;
typedef CGAL::Nth_of_tuple_property_map<11, GaussianTuple> Rot_3_map;

struct GaussianSplat {
  double x, y, z;
  double nx, ny, nz;
  double f_dc_0, f_dc_1, f_dc_2;
  double scale_0, scale_1, scale_2;
  double rot_0, rot_1, rot_2, rot_3;
};

struct Rectangle3D {
  Point_3 center;
  Vector_3 axis1, axis2;
  double width, height;
  double r, g, b;
};

// Function to transform from OpenCV to CGAL coordinate system
// -90 degree rotation around X-axis: Y -> -Z, Z -> Y
void transform_opencv_to_cgal(Point_3 &p) {
  double temp_y = p.y();
  double temp_z = p.z();
  // X stays the same
  p = Point_3(p.x(), -temp_z, temp_y);
}

// Function to read PLY file using CGAL
std::vector<GaussianSplat> read_ply(const std::string& filename) {
  std::vector<GaussianSplat> gaussians;

  std::cout << "Reading PLY file with CGAL..." << std::endl;
  
  // Store points as tuples
  std::vector<GaussianTuple> points;
  
  // Read PLY file with properties
  std::ifstream in(filename, std::ios::binary);
  if (!in) { throw std::runtime_error("Cannot open file: " + filename); }
  bool success = CGAL::IO::read_PLY_with_properties(
    in, std::back_inserter(points),
    CGAL::IO::make_ply_point_reader(Point_map()),
    CGAL::IO::make_ply_normal_reader(Normal_map()),
    std::make_pair(F_dc_0_map(), CGAL::IO::PLY_property<double>("f_dc_0")),
    std::make_pair(F_dc_1_map(), CGAL::IO::PLY_property<double>("f_dc_1")),
    std::make_pair(F_dc_2_map(), CGAL::IO::PLY_property<double>("f_dc_2")),
    std::make_pair(Scale_0_map(), CGAL::IO::PLY_property<double>("scale_0")),
    std::make_pair(Scale_1_map(), CGAL::IO::PLY_property<double>("scale_1")),
    std::make_pair(Scale_2_map(), CGAL::IO::PLY_property<double>("scale_2")),
    std::make_pair(Rot_0_map(), CGAL::IO::PLY_property<double>("rot_0")),
    std::make_pair(Rot_1_map(), CGAL::IO::PLY_property<double>("rot_1")),
    std::make_pair(Rot_2_map(), CGAL::IO::PLY_property<double>("rot_2")),
    std::make_pair(Rot_3_map(), CGAL::IO::PLY_property<double>("rot_3"))
  );
  if (!success) { throw std::runtime_error("Failed to read PLY file: " + filename); }
  std::cout << "Successfully read " << points.size() << " points from PLY" << std::endl;
  
  // Convert tuples to GaussianSplat structs
  gaussians.reserve(points.size());
  for (const auto& pt : points) {
    GaussianSplat gs;
    
    // Extract position
    Point_3 pos = std::get<0>(pt);
    gs.x = pos.x();
    gs.y = pos.y();
    gs.z = pos.z();
    
    // Extract normal
    Vector_3 normal = std::get<1>(pt);
    gs.nx = normal.x();
    gs.ny = normal.y();
    gs.nz = normal.z();
    
    // Extract other properties
    gs.f_dc_0 = std::get<2>(pt);
    gs.f_dc_1 = std::get<3>(pt);
    gs.f_dc_2 = std::get<4>(pt);
    gs.scale_0 = std::get<5>(pt);
    gs.scale_1 = std::get<6>(pt);
    gs.scale_2 = std::get<7>(pt);
    gs.rot_0 = std::get<8>(pt);
    gs.rot_1 = std::get<9>(pt);
    gs.rot_2 = std::get<10>(pt);
    gs.rot_3 = std::get<11>(pt);
    
    gaussians.push_back(gs);
  }
  
  return gaussians;
}

// Function to normalize quaternion
void normalize_quaternion(double& w, double& x, double& y, double& z) {
  double norm = std::sqrt(w*w + x*x + y*y + z*z);
  if (norm > 1e-10) {
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;
  }
}

// Function to convert quaternion to rotation matrix
void quat_to_matrix(double w, double x, double y, double z, double R[3][3]) {
  R[0][0] = 1 - 2*y*y - 2*z*z;
  R[0][1] = 2*x*y - 2*w*z;
  R[0][2] = 2*x*z + 2*w*y;
  
  R[1][0] = 2*x*y + 2*w*z;
  R[1][1] = 1 - 2*x*x - 2*z*z;
  R[1][2] = 2*y*z - 2*w*x;
  
  R[2][0] = 2*x*z - 2*w*y;
  R[2][1] = 2*y*z + 2*w*x;
  R[2][2] = 1 - 2*x*x - 2*y*y;
}

// Function to convert Gaussian splat to 3D rectangle
Rectangle3D gs_to_rect(const GaussianSplat& gs) {
  Rectangle3D rect;
  
  rect.center = Point_3(gs.x, gs.y, gs.z);
  
  // Get scales and find two largest axes
  std::array<double, 3> scales = {std::exp(gs.scale_0), std::exp(gs.scale_1), std::exp(gs.scale_2)};
  std::array<int, 3> indices = {0, 1, 2};
  
  std::sort(indices.begin(), indices.end(), 
        [&scales](int a, int b) { return scales[a] > scales[b]; });
  
  // Normalize quaternion and convert to rotation matrix
  double w = gs.rot_0, x = gs.rot_1, y = gs.rot_2, z = gs.rot_3;
  normalize_quaternion(w, x, y, z);
  
  double R[3][3];
  quat_to_matrix(w, x, y, z, R);
  
  // Get the two principal axes
  Vector_3 axis1(R[0][indices[0]], R[1][indices[0]], R[2][indices[0]]);
  Vector_3 axis2(R[0][indices[1]], R[1][indices[1]], R[2][indices[1]]);
  
  rect.axis1 = axis1;
  rect.axis2 = axis2;
  rect.width = 2.0 * scales[indices[0]];
  rect.height = 2.0 * scales[indices[1]];
  
  // Convert color
  rect.r = std::max(0.0, std::min(1.0, (gs.f_dc_0 + 0.5)));
  rect.g = std::max(0.0, std::min(1.0, (gs.f_dc_1 + 0.5)));
  rect.b = std::max(0.0, std::min(1.0, (gs.f_dc_2 + 0.5)));
  
  return rect;
}

// Function to add rectangle to mesh with colors
void add_rectangle_to_mesh(Mesh& mesh, const Rectangle3D& rect) {
  Vector_3 half_width_vec = (rect.width / 2.0) * rect.axis1;
  Vector_3 half_height_vec = (rect.height / 2.0) * rect.axis2;
  
  Point_3 p1 = rect.center + half_width_vec + half_height_vec;
  Point_3 p2 = rect.center - half_width_vec + half_height_vec;
  Point_3 p3 = rect.center - half_width_vec - half_height_vec;
  Point_3 p4 = rect.center + half_width_vec - half_height_vec;

  transform_opencv_to_cgal(p1);
  transform_opencv_to_cgal(p2);
  transform_opencv_to_cgal(p3);
  transform_opencv_to_cgal(p4);
  
  auto v1 = mesh.add_vertex(p1);
  auto v2 = mesh.add_vertex(p2);
  auto v3 = mesh.add_vertex(p3);
  auto v4 = mesh.add_vertex(p4);
  
  auto f1 = mesh.add_face(v1, v2, v3);
  auto f2 = mesh.add_face(v1, v3, v4);
  
  // Add color property to mesh if it doesn't exist
  auto color_map_opt = mesh.property_map<Mesh::Face_index, CGAL::IO::Color>("f:color");
  if (!color_map_opt) {
    throw std::runtime_error("Color property map not found");
  }
  auto color_map = *color_map_opt;
  
  // Convert to CGAL Color (RGB values 0-255)
  CGAL::IO::Color color(
    (unsigned char)(rect.r * 255),
    (unsigned char)(rect.g * 255), 
    (unsigned char)(rect.b * 255)
  );
  
  // Assign color to both faces of this rectangle
  color_map[f1] = color;
  color_map[f2] = color;
}

// Function to write colored rectangles to PLY file
void write_ply(const std::vector<Rectangle3D>& rectangles, const std::string& filename) {
  Mesh mesh;
  // Add color property to mesh
  auto color_map = mesh.add_property_map<Mesh::Face_index, CGAL::IO::Color>("f:color", CGAL::IO::Color(128, 128, 128)).first;
  
  for (const auto& rect : rectangles) {
    add_rectangle_to_mesh(mesh, rect);
  }
  
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot create output file: " + filename);
  }
  
  // Write PLY with colors
  CGAL::IO::write_PLY(out, mesh, CGAL::parameters::face_color_map(color_map));
  out.close();
  
  std::cout << "Written " << rectangles.size() << " colored rectangles (" 
            << mesh.number_of_faces() << " faces, " 
            << mesh.number_of_vertices() << " vertices) to " << filename << "\n";
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply>\n";
    return 1;
  }
  
  try {
    std::cout << "Reading Gaussian splats from " << argv[1] << "...\n";
    auto gaussians = read_ply(argv[1]);
    
    std::cout << "Converting " << gaussians.size() << " Gaussians to rectangles...\n";
    std::vector<Rectangle3D> rectangles;
    rectangles.reserve(gaussians.size());
    
    for (const auto& gs : gaussians) {
      rectangles.push_back(gs_to_rect(gs));
    }
    
    std::cout << "Writing rectangles to " << argv[2] << "...\n";
    write_ply(rectangles, argv[2]);
    
    std::cout << "Conversion completed!\n";
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}