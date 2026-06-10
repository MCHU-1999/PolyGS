#include "include/Kinetic_surface_reconstruction_3.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <filesystem>
#include <iostream>
#include <string>

#include <CGAL/mst_orient_normals.h>
#include <CGAL/pca_estimate_normals.h>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;
using Segment_3 = typename Kernel::Segment_3;

using Point_set = CGAL::Point_set_3<Point_3>;
using Point_map = typename Point_set::Point_map;
using Normal_map = typename Point_set::Vector_map;

using KSR = CGAL::Kinetic_surface_reconstruction_3<Kernel, Point_set, Point_map,
                                                   Normal_map>;

int main(int argc, char **argv) {
  // Input and CLI args.
  std::string input_file, output_dir;
  auto print_usage = [&](const char *prog) {
    std::cout << "Usage: " << prog
              << " [-i|--input <input.ply>] [-o|--output <output_dir>]"
              << std::endl;
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (a == "-i" || a == "--input") {
      if (i + 1 < argc)
        input_file = argv[++i];
      else {
        std::cerr << "Error: missing argument for " << a << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
    } else if (a == "-o" || a == "--output") {
      if (i + 1 < argc)
        output_dir = argv[++i];
      else {
        std::cerr << "Error: missing argument for " << a << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
    } else {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  namespace fs = std::filesystem;
  if (!fs::exists(input_file)) {
    std::cerr << "Input file not found: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  fs::path outdir(output_dir);
  std::error_code ec;
  if (!fs::exists(outdir)) {
    if (!fs::create_directories(outdir, ec)) {
      std::cerr << "Failed to create output directory '" << output_dir
                << "': " << ec.message() << std::endl;
      return EXIT_FAILURE;
    }
  } else if (!fs::is_directory(outdir)) {
    std::cerr << "Output path exists and is not a directory: " << output_dir
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading input: " << input_file << std::endl;
  std::cout << "Writing outputs to: " << outdir << std::endl;

  Point_set point_set;
  auto assignment_prop =
      point_set.add_property_map<int>("pts_ins_assignment", 0).first;

  if (!CGAL::IO::read_point_set(input_file, point_set)) {
    std::cerr << "Failed to read point set from: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  bool need_normals = false;
  if (point_set.has_normal_map() && point_set.begin() != point_set.end()) {
    auto n = point_set.normal(*point_set.begin());
    if (n.squared_length() < 1e-6) {
      need_normals = true;
    }
  } else {
    need_normals = true;
  }

  if (need_normals) {
    std::cout
        << "Normals are missing or zero. Estimating and orienting normals..."
        << std::endl;
    if (!point_set.has_normal_map()) {
      point_set.add_normal_map();
    }
    CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
        point_set, 12,
        point_set.parameters()
            .point_map(point_set.point_map())
            .normal_map(point_set.normal_map()));
    CGAL::mst_orient_normals(point_set, 12,
                             point_set.parameters()
                                 .point_map(point_set.point_map())
                                 .normal_map(point_set.normal_map()));
  }

  bool has_assignment = false;
  std::vector<int> plane_assignments;
  plane_assignments.reserve(point_set.size());
  for (auto it = point_set.begin(); it != point_set.end(); ++it) {
    int val = assignment_prop[*it];
    plane_assignments.push_back(val);
    if (val > 0) {
      has_assignment = true;
    }
  }

  std::map<typename KSR::KSP::Face_support, bool> external_nodes;
  // All bbox faces prefer "outside" label except YMAX (intentional for model orientation).
  external_nodes[KSR::KSP::Face_support::ZMIN] = false;
  external_nodes[KSR::KSP::Face_support::ZMAX] = false;
  external_nodes[KSR::KSP::Face_support::XMIN] = false;
  external_nodes[KSR::KSP::Face_support::XMAX] = false;
  external_nodes[KSR::KSP::Face_support::YMIN] = false;
  external_nodes[KSR::KSP::Face_support::YMAX] = true;
 
  auto param =CGAL::parameters::k_neighbors(8)
    // Octree controls: suppress axis-aligned octree split planes (Face_support::OCTREE_FACE = -7)
    // that appear in output when #planes > max_octree_node_size. Raise threshold to avoid splits.
    // .max_octree_depth(1)          // at most 1 level of octree subdivision
    // .max_octree_node_size(10000)  // only split if >10000 polygons per node (effectively disables splits)
    // Regularization: applies to both inject_planar_shapes and detect_planar_shapes paths.
    .regularize_parallelism(true)
    .regularize_coplanarity(true)
    .regularize_orthogonality(true)
    .angle_tolerance(15)
    .maximum_offset(0.05); // maximum distance between two parallel planes to be coplanar


  // Algorithm.
  KSR ksr(point_set, param);
  if (has_assignment) {
    std::cout
        << "Found pts_ins_assignment property. Using external plane detections."
        << std::endl;
    std::cout << "  Point set size: " << point_set.size() << std::endl;
    std::cout << "  Plane assignments size: " << plane_assignments.size()
              << std::endl;
    int max_label = 0;
    for (int v : plane_assignments)
      max_label = std::max(max_label, v);
    std::cout << "  Max plane label: " << max_label << std::endl;
    ksr.injection_and_partition(plane_assignments, 2, param);
    std::cout << "injection_and_partition completed." << std::endl;
    std::cout << "  Number of volumes: "
              << ksr.kinetic_partition().number_of_volumes() << std::endl;
    // Diagnostic: how many planes did KSP actually receive?
    // If less than max_label, coplanar planes were deduplicated by KSP.
    // If output contains axis-aligned phantom planes, octree subdivision is the cause.
    std::cout << "  KSP input_planes count: "
              << ksr.kinetic_partition().input_planes().size()
              << " (injected " << max_label << " labeled planes)" << std::endl;
  } else {
    std::cout << "No pts_ins_assignment property found. Falling back to "
                 "internal CGAL shape detection."
              << std::endl;
    ksr.detection_and_partition(3, param);
  }

  std::vector<Point_3> vtx;
  std::vector<std::vector<std::size_t>> polylist;
  std::vector<FT> lambdas{0.1, 0.3, 0.5, 0.7, 0.9};

  bool non_empty = false;
  for (FT l : lambdas) {
    vtx.clear();
    polylist.clear();
    // std::cout << "Reconstructing with lambda=" << CGAL::to_double(l) << "..." << std::endl;
    // ksr.reconstruct_with_ground(l, std::back_inserter(vtx), std::back_inserter(polylist));
    ksr.reconstruct(l, external_nodes, std::back_inserter(vtx), std::back_inserter(polylist));
    
    std::cout << "  => vtx=" << vtx.size() << " polylist=" << polylist.size() << std::endl;

    if (polylist.size() > 0) {
      non_empty = true;

      // Repair the soup: removes duplicates and degenerated faces
      CGAL::Polygon_mesh_processing::repair_polygon_soup(vtx, polylist);
      // Orient the soup: fixes inconsistent normals which cause non-manifold errors
      CGAL::Polygon_mesh_processing::orient_polygon_soup(vtx, polylist);

      std::string lstr = std::to_string(CGAL::to_double(l));
      std::string filename = "polylist_" + lstr + ".ply";
      fs::path outp = outdir / filename;
      bool success = CGAL::IO::write_polygon_soup(outp.string(), vtx, polylist);
      if (success) {
        std::cout << "Wrote " << outp << std::endl;
      } else {
        std::cout << "Failed to write " << outp << std::endl;
      }
    }
  }

  return (non_empty) ? EXIT_SUCCESS : EXIT_FAILURE;
}