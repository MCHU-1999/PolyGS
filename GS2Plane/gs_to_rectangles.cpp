// #include <CGAL/Simple_cartesian.h>
// #include <CGAL/Surface_mesh.h>
// #include "custom_ds.h"
#include "custom_region_growing.h"
// #include "custom_region_growing_2.h"
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
#include <random>

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
  rect.normal = Vector_3(gs.nx, gs.ny, gs.nz);
  
  // Get scales and find two largest axes
  std::array<double, 3> scales = {std::exp(gs.scale_0), std::exp(gs.scale_1), std::exp(gs.scale_2)};
  std::array<int, 3> indices = {0, 1, 2};
  
  std::sort(indices.begin(), indices.end(), 
        [&scales](int a, int b) { return abs(scales[a]) > abs(scales[b]); });
  
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
  rect.width = 2.0 * abs(scales[indices[0]]);
  rect.height = 2.0 * abs(scales[indices[1]]);
  rect.pseudo_radius = (rect.width + rect.height) / 4;
  
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

std::array<double, 3> hue_to_rgb(double hue_fraction, double saturation = 0.8, double brightness = 0.9) {
  // Convert hue fraction (0-1) to degrees (0-360)
  double hue_deg = hue_fraction * 360.0;
  
  // Simple HSV to RGB conversion
  double chroma = brightness * saturation;
  double hue_prime = hue_deg / 60.0;
  double x = chroma * (1.0 - std::abs(std::fmod(hue_prime, 2.0) - 1.0));
  double m = brightness - chroma;
  
  double r, g, b;
  if (hue_prime < 1) {
    r = chroma; g = x; b = 0;
  } else if (hue_prime < 2) {
    r = x; g = chroma; b = 0;
  } else if (hue_prime < 3) {
    r = 0; g = chroma; b = x;
  } else if (hue_prime < 4) {
    r = 0; g = x; b = chroma;
  } else if (hue_prime < 5) {
    r = x; g = 0; b = chroma;
  } else {
    r = chroma; g = 0; b = x;
  }
  
  return {r + m, g + m, b + m};
}

// Function to create cluster-based colors with better distribution
std::vector<std::array<double, 3>> create_cluster_colors(size_t num_clusters) {
  std::vector<std::array<double, 3>> colors(num_clusters);
  
  // Use golden ratio for better color distribution
  const double golden_ratio = 0.618033988749895;
  
  for (size_t i = 0; i < num_clusters; ++i) {
    // Distribute hues evenly using golden ratio
    double hue_fraction = std::fmod(i * golden_ratio, 1.0);
    
    // Vary saturation and brightness slightly to add more distinction
    double saturation = 0.7 + 0.3 * std::sin(i * 0.5);  // 0.7 to 1.0
    double brightness = 0.8 + 0.2 * std::cos(i * 0.7);  // 0.8 to 1.0
    
    colors[i] = hue_to_rgb(hue_fraction, saturation, brightness);
  }
  
  return colors;
}

// Function to apply cluster colors to rectangles
void color_rectangles_by_cluster(std::vector<Rectangle3D>& rectangles,
                                const std::vector<std::vector<std::size_t>>& clusters) {
  
  // Generate unique colors for each cluster
  auto cluster_colors = create_cluster_colors(clusters.size());
  
  // Default gray color for unclustered rectangles
  const std::array<double, 3> gray_color = {0.5, 0.5, 0.5};
  
  // First, color ALL rectangles gray (unclustered)
  for (auto& rect : rectangles) {
    rect.r = gray_color[0];
    rect.g = gray_color[1];
    rect.b = gray_color[2];
  }
  
  std::cout << "Coloring " << clusters.size() << " clusters..." << std::endl;
  
  // Then, apply colors to each cluster (overriding gray for clustered rectangles)
  for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
    const auto& cluster = clusters[cluster_idx];
    const auto& color = cluster_colors[cluster_idx];
    
    std::cout << "Cluster " << cluster_idx << ": " << cluster.size() 
              << " rectangles, RGB(" 
              << (int)(color[0]*255) << ", " 
              << (int)(color[1]*255) << ", " 
              << (int)(color[2]*255) << ")" << std::endl;
    
    // Color all rectangles in this cluster
    for (auto rect_idx : cluster) {
      if (rect_idx < rectangles.size()) {
        rectangles[rect_idx].r = color[0];
        rectangles[rect_idx].g = color[1];
        rectangles[rect_idx].b = color[2];
      }
    }
  }
  
  // Count how many rectangles are unclustered
  std::vector<bool> is_clustered(rectangles.size(), false);
  for (const auto& cluster : clusters) {
    for (auto rect_idx : cluster) {
      if (rect_idx < rectangles.size()) {
        is_clustered[rect_idx] = true;
      }
    }
  }
  
  size_t unclustered_count = std::count(is_clustered.begin(), is_clustered.end(), false);
  std::cout << "Unclustered rectangles: " << unclustered_count << " (colored gray)" << std::endl;
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

    // // Take every 10th rectangle to get a representative sample
    // std::vector<Rectangle3D> test_rectangles;
    // for (size_t i = 0; i < rectangles.size(); i += 10) {
    //   test_rectangles.push_back(rectangles[i]);
    // }
    // std::cout << "Testing with " << test_rectangles.size() << " rectangles (sampled from " << rectangles.size() << ")" << std::endl;

    // Detect planar regions
    std::cout << "Detecting planar regions...\n";
    auto detected_regions = detect_planar_regions(
      rectangles,
      0.20,
      0.95,
      0.05,
      100
    );
    if (detected_regions.empty()) {
      std::cerr << "No clusters found" << std::endl;
      return EXIT_FAILURE;
    }

    // Color rectangles based on their cluster membership
    color_rectangles_by_cluster(rectangles, detected_regions);
    
    // Write colored output
    std::cout << "Writing colored rectangles to " << argv[2] << "...\n";
    write_ply(rectangles, argv[2]);
    std::cout << "Conversion completed!\n";
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}