#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/compute_average_spacing.h>
#include <iostream>
#include <vector>
#include <filesystem> // C++17 for path handling

namespace fs = std::filesystem;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_with_normal = std::pair<Point_3, Vector_3>;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;

int main(int argc, char** argv) {
    std::string input_file = (argc > 1) ? argv[1] : "input.ply";
    std::string output_file;

    if (argc > 2) {
        output_file = argv[2];
    } else {
        // Extract input directory and append default filename
        fs::path p(input_file);
        output_file = (p.parent_path() / "output_poisson.ply").string();
    }

    std::vector<Point_with_normal> points;
    std::cout << "Reading " << input_file << "..." << std::endl;
    
    if(!CGAL::IO::read_points(input_file, std::back_inserter(points),
                              CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>())
                                              .normal_map(CGAL::Second_of_pair_property_map<Point_with_normal>()))) {
        std::cerr << "Error: Cannot read PLY file or normals missing." << std::endl;
        return 1;
    }

    double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
        points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Point_with_normal>())
    );

    Surface_mesh output_mesh;
    if (CGAL::poisson_surface_reconstruction_delaunay(
            points.begin(), points.end(),
            CGAL::First_of_pair_property_map<Point_with_normal>(),
            CGAL::Second_of_pair_property_map<Point_with_normal>(),
            output_mesh, average_spacing)) 
    {
        std::cout << "Saving to: " << output_file << std::endl;
        CGAL::IO::write_polygon_mesh(output_file, output_mesh);
    }

    return 0;
}