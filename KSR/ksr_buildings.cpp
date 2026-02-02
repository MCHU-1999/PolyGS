#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kinetic_surface_reconstruction_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <filesystem>
#include <iostream>
#include <string>

using Kernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT        = typename Kernel::FT;
using Point_3   = typename Kernel::Point_3;
using Vector_3  = typename Kernel::Vector_3;
using Segment_3 = typename Kernel::Segment_3;
 
using Point_set    = CGAL::Point_set_3<Point_3>;
using Point_map    = typename Point_set::Point_map;
using Normal_map   = typename Point_set::Vector_map;
 
using KSR = CGAL::Kinetic_surface_reconstruction_3<Kernel, Point_set, Point_map, Normal_map>;
 
int main(int argc, char** argv) {
  // Input and CLI args.
  std::string input_file, output_dir;
  auto print_usage = [&](const char* prog){
    std::cout << "Usage: " << prog << " [-i|--input <input.ply>] [-o|--output <output_dir>]" << std::endl;
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { print_usage(argv[0]); return EXIT_SUCCESS; }
    else if (a == "-i" || a == "--input") {
      if (i + 1 < argc) input_file = argv[++i]; else { std::cerr << "Error: missing argument for " << a << std::endl; print_usage(argv[0]); return EXIT_FAILURE; }
    } else if (a == "-o" || a == "--output") {
      if (i + 1 < argc) output_dir = argv[++i]; else { std::cerr << "Error: missing argument for " << a << std::endl; print_usage(argv[0]); return EXIT_FAILURE; }
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
      std::cerr << "Failed to create output directory '" << output_dir << "': " << ec.message() << std::endl;
      return EXIT_FAILURE;
    }
  } else if (!fs::is_directory(outdir)) {
    std::cerr << "Output path exists and is not a directory: " << output_dir << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading input: " << input_file << std::endl;
  std::cout << "Writing outputs to: " << outdir << std::endl;

  Point_set point_set;
  if (!CGAL::IO::read_point_set(input_file, point_set)) {
    std::cerr << "Failed to read point set from: " << input_file << std::endl;
    return EXIT_FAILURE;
  }
 
  auto param = CGAL::parameters::maximum_distance(0.01)  // the maximum distance from a point to a plane
    .maximum_angle(5)               // the maximum angle in degrees between the normal associated with a point and the normal of a plane
    .minimum_region_size(10)        // the minimum number of points a region must have
    .reorient_bbox(false)           // Setting reorient_bbox to true aligns the x-axis of the bounding box with the direction of the largest variation in horizontal direction of the input data while maintaining the z-axis.
    .regularize_parallelism(true)   // whether parallelism should be regularized or not
    .regularize_coplanarity(true)   // whether coplanarity should be regularized or not
    .angle_tolerance(5)             // Idk
    .maximum_offset(0.01);          // maximum distance between two parallel planes to be considered coplanar

  // Algorithm.
  KSR ksr(point_set, param);
 
  ksr.detection_and_partition(2, param);
 
  std::vector<Point_3> vtx;
  std::vector<std::vector<std::size_t> > polylist;
 
  std::vector<FT> lambdas{0.5, 0.7, 0.8, 0.9};
 
  bool non_empty = false;
 
  for (FT l : lambdas) {
    vtx.clear();
    polylist.clear();
 
    ksr.reconstruct(l, std::back_inserter(vtx), std::back_inserter(polylist));
 
    if (polylist.size() > 0) {
      non_empty = true;
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