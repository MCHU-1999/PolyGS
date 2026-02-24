#include "cp_d0_dist_wrapper.h"
#include "gs_to_graph.h"
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
typedef CGAL::Nth_of_tuple_property_map<12, GaussianTuple> Opacity_map;

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
    std::make_pair(Rot_3_map(), CGAL::IO::PLY_property<double>("rot_3")),
    std::make_pair(Opacity_map(), CGAL::IO::PLY_property<double>("opacity"))
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
    gs.opacity = std::get<12>(pt);
    
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
  rect.height = 2.0 * abs(scales[indices[0]]);
  rect.width = 2.0 * abs(scales[indices[1]]);
  rect.pseudo_radius = CGAL::sqrt(std::pow(rect.width, 2) + std::pow(rect.height, 2));

  Vector_3 half_height_vec = (rect.height / 2.0) * rect.axis1;
  Vector_3 half_width_vec = (rect.width / 2.0) * rect.axis2;
  rect.v1 = rect.center + half_width_vec + half_height_vec;
  rect.v2 = rect.center - half_width_vec + half_height_vec;
  rect.v3 = rect.center - half_width_vec - half_height_vec;
  rect.v4 = rect.center + half_width_vec - half_height_vec;
  
  // Convert color
  rect.red = std::max(0.0, std::min(1.0, (gs.f_dc_0 + 0.5)));
  rect.green = std::max(0.0, std::min(1.0, (gs.f_dc_1 + 0.5)));
  rect.blue = std::max(0.0, std::min(1.0, (gs.f_dc_2 + 0.5)));
  
  return rect;
}

// Function to add rectangle to mesh with colors
void add_rectangle_to_mesh(Mesh& mesh, const Rectangle3D& rect) {
  Point_3 p1(rect.v1);
  Point_3 p2(rect.v2);
  Point_3 p3(rect.v3);
  Point_3 p4(rect.v4);
  // transform_opencv_to_cgal(p1);
  // transform_opencv_to_cgal(p2);
  // transform_opencv_to_cgal(p3);
  // transform_opencv_to_cgal(p4);
  
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
    (unsigned char)(rect.red * 255),
    (unsigned char)(rect.green * 255), 
    (unsigned char)(rect.blue * 255)
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
// Templated to accept std::vector<std::vector<int>>, std::vector<std::vector<size_t>>, etc.
template <typename IndexType>
void color_rectangles_by_cluster(std::vector<Rectangle3D>& rectangles,
                                const std::vector<std::vector<IndexType>>& clusters) {
  
  // Generate unique colors for each cluster
  auto cluster_colors = create_cluster_colors(clusters.size());
  
  // Default gray color for unclustered rectangles (or init with gray)
  const std::array<double, 3> gray_color = {0.5, 0.5, 0.5};
  
  // First, color ALL rectangles gray (unclustered)
  for (auto& rect : rectangles) {
    rect.red = gray_color[0];
    rect.green = gray_color[1];
    rect.blue = gray_color[2];
  }
  
  std::cout << "Coloring " << clusters.size() << " clusters..." << std::endl;
  
  // Then, apply colors to each cluster
  for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
    const auto& cluster = clusters[cluster_idx];
    const auto& color = cluster_colors[cluster_idx];
    
    // Optional: Print info only for larger clusters to avoid spam
    if (cluster_idx < 10 || cluster.size() > 100) {
      std::cout << "Cluster " << cluster_idx << ": " << cluster.size() 
                << " rectangles." << std::endl;
    }
    
    // Color all rectangles in this cluster
    for (auto rect_idx : cluster) {
      // Cast index to size_t to ensure safe comparison/indexing
      size_t idx = static_cast<size_t>(rect_idx);
      
      if (idx < rectangles.size()) {
        rectangles[idx].red = color[0];
        rectangles[idx].green = color[1];
        rectangles[idx].blue = color[2];
      }
    }
  }
  
  // Verify coverage
  size_t colored_count = 0;
  for (const auto& cluster : clusters) {
      colored_count += cluster.size();
  }
  
  std::cout << "Total colored: " << colored_count << "/" << rectangles.size() << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply>\n";
    return 1;
  }

  const std::vector<double> WEIGHTS = {
    20, 20, 20,
    30, 30, 30,
    1, 1, 1,
    0, 0, 0,
    0, 0, 0, 0,
    0
  };
  constexpr int D = 17; 
  
  try {
    std::cout << "Reading Gaussian splats from " << argv[1] << "...\n";
    auto gaussians = read_ply(argv[1]);
    
    std::cout << "Converting " << gaussians.size() << " Gaussians to rectangles...\n";
    std::vector<Rectangle3D> rectangles;
    rectangles.reserve(gaussians.size());
    
    for (const auto& gs : gaussians) {
      rectangles.push_back(gs_to_rect(gs));
    }

    std::cout << "Building graph...\n";
    // ForwardStarGraph fsg = create_forward_star(rectangles, 0.3, 0.0, 0.02, 0.9);
    ForwardStarGraph fsg = create_forward_star_knn(rectangles, 10);

    std::cout << "Calculating edge weights..." << std::endl;
    double sum_dist = 0.0;
    for (double d : fsg.distances) {
      sum_dist += d;
    }
    double eps = 1e-12;
    double mean_dist = (fsg.distances.empty()) ? 1.0 : (sum_dist / fsg.distances.size()) + eps;
    std::vector<double> edge_weights;
    edge_weights.reserve(fsg.distances.size());
    for (double d : fsg.distances) {
      edge_weights.push_back(std::exp(-d / mean_dist));
    }

    // Construct features from Gaussians
    std::cout << "Constructing features (D=" << D << ")...\n";
    std::vector<std::array<double, D>> features;
    features.reserve(gaussians.size());
    for (const auto& gs : gaussians) {
      features.push_back({
        gs.x, gs.y, gs.z,                                 // 0-2: Pos
        gs.nx, gs.ny, gs.nz,                              // 3-5: Normal
        gs.f_dc_0, gs.f_dc_1, gs.f_dc_2,                  // 6-8: Color
        gs.scale_0, gs.scale_1, gs.scale_2,               // 9-11: Scale
        gs.rot_0, gs.rot_1, gs.rot_2, gs.rot_3,           // 12-15: Rot
        gs.opacity                                        // 16: Opacity
      });
    }
    
    // Configure Cut Pursuit parameters
    CutPursuitParams<double> cp_params;
    cp_params.coor_weights = WEIGHTS;
    cp_params.edge_weights = edge_weights;
    cp_params.loss = D;
    cp_params.min_comp_weight = 20;
    // cp_params.max_split_size = 400;
    cp_params.cp_it_max = 20;
    cp_params.K = 2;
    cp_params.cp_dif_tol = 1e-2;
    cp_params.split_damp_ratio = 0.7;
    cp_params.verbose = 1;
    cp_params.balance_parallel_split=true;
    cp_params.compute_Time=true;
    cp_params.compute_List=true;
    cp_params.compute_Graph=true;

    std::cout << "Running Cut Pursuit...\n";
    auto results = cut_pursuit_points<double, D>(
      features,
      fsg.first_edge,
      fsg.adj_vertices,
      cp_params
    );

    auto& partitions = results.component_lists;
    // Color rectangles based on their cluster membership
    color_rectangles_by_cluster(rectangles, partitions);
    
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