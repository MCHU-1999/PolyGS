#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/IO/read_points.h>
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;

int main(int argc, char** argv) {
    std::string input_file = (argc > 1) ? argv[1] : "input.ply";
    std::string output_file;

    if (argc > 2) {
        output_file = argv[2];
    } else {
        fs::path p(input_file);
        output_file = (p.parent_path() / "output_alphawrap.ply").string();
    }

    std::vector<Point_3> points;
    if(!CGAL::IO::read_points(input_file, std::back_inserter(points))) {
        std::cerr << "Error: Cannot read PLY file." << std::endl;
        return 1;
    }

    CGAL::Bbox_3 bbox = CGAL::bbox_3(points.begin(), points.end());
    double diag = std::sqrt(std::pow(bbox.xmax()-bbox.xmin(), 2) + 
                            std::pow(bbox.ymax()-bbox.ymin(), 2) + 
                            std::pow(bbox.zmax()-bbox.zmin(), 2));

    // alpha = diag / 50.0 is a common starting point for "tight" wraps
    double alpha = diag / 500.0;
    double offset = diag / 2000.0;

    Surface_mesh wrap;
    std::cout << "Wrapping points..." << std::endl;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);

    std::cout << "Saving to: " << output_file << std::endl;
    CGAL::IO::write_polygon_mesh(output_file, wrap);

    return 0;
}