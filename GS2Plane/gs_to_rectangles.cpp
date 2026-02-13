#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

struct GaussianSplat {
    // Position
    double x, y, z;
    
    // Normal vectors (might not be used directly)
    double nx, ny, nz;
    
    // Color
    double f_dc_0, f_dc_1, f_dc_2;
    
    // Scale (ellipsoid semi-axes)
    double scale_0, scale_1, scale_2;
    
    // Rotation quaternion
    double rot_0, rot_1, rot_2, rot_3;  // w, x, y, z
    
    // Opacity
    double opacity;
};

struct Rectangle3D {
    Point_3 center;
    Vector_3 axis1, axis2;  // The two main axes
    double width, height;   // Dimensions along the two main axes
    double r, g, b;        // Color
    double opacity;
};

// Function to read PLY file and extract Gaussian splats
std::vector<GaussianSplat> read_ply(const std::string& filename) {
    std::vector<GaussianSplat> gaussians;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    bool in_header = true;
    int vertex_count = 0;
    std::vector<std::string> properties;
    
    // Parse header
    while (std::getline(file, line) && in_header) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string element, vertex;
            iss >> element >> vertex >> vertex_count;
        }
        else if (line.find("property") != std::string::npos) {
            std::istringstream iss(line);
            std::string property, type, name;
            iss >> property >> type >> name;
            properties.push_back(name);
        }
        else if (line == "end_header") {
            in_header = false;
        }
    }
    
    // Read vertex data
    gaussians.reserve(vertex_count);
    for (int i = 0; i < vertex_count; ++i) {
        if (!std::getline(file, line)) break;
        
        std::istringstream iss(line);
        GaussianSplat gs;
        
        // Map property values to struct members
        std::vector<double> values;
        double val;
        while (iss >> val) {
            values.push_back(val);
        }
        
        // Assuming the properties are in the expected order
        // You might need to adjust indices based on actual PLY structure
        int idx = 0;
        gs.x = values[idx++];
        gs.y = values[idx++];
        gs.z = values[idx++];
        gs.nx = values[idx++];
        gs.ny = values[idx++];
        gs.nz = values[idx++];
        gs.f_dc_0 = values[idx++];
        gs.f_dc_1 = values[idx++];
        gs.f_dc_2 = values[idx++];
        gs.scale_0 = values[idx++];
        gs.scale_1 = values[idx++];
        gs.scale_2 = values[idx++];
        gs.rot_0 = values[idx++];  // w
        gs.rot_1 = values[idx++];  // x
        gs.rot_2 = values[idx++];  // y
        gs.rot_3 = values[idx++];  // z
        gs.opacity = values[idx++];
        
        gaussians.push_back(gs);
    }
    
    std::cout << "Loaded " << gaussians.size() << " Gaussian splats\n";
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
    normalize_quaternion(w, x, y, z);
    
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
    
    // Set center position
    rect.center = Point_3(gs.x, gs.y, gs.z);
    
    // Get scales and find two largest axes
    std::array<double, 3> scales = {std::exp(gs.scale_0), std::exp(gs.scale_1), std::exp(gs.scale_2)};
    std::array<int, 3> indices = {0, 1, 2};
    
    // Sort indices by scale values (descending)
    std::sort(indices.begin(), indices.end(), 
              [&scales](int a, int b) { return scales[a] > scales[b]; });
    
    // Convert quaternion to rotation matrix
    double R[3][3];
    quat_to_matrix(gs.rot_0, gs.rot_1, gs.rot_2, gs.rot_3, R);
    
    // Get the two principal axes (corresponding to largest scales)
    Vector_3 axis1(R[0][indices[0]], R[1][indices[0]], R[2][indices[0]]);
    Vector_3 axis2(R[0][indices[1]], R[1][indices[1]], R[2][indices[1]]);
    
    rect.axis1 = axis1;
    rect.axis2 = axis2;
    rect.width = 2.0 * scales[indices[0]];   // Full width along first axis
    rect.height = 2.0 * scales[indices[1]];  // Full height along second axis
    
    // Convert color from spherical harmonics to RGB
    // Simple approximation: f_dc_0/1/2 are already close to RGB
    rect.r = std::max(0.0, std::min(1.0, (gs.f_dc_0 + 1.0) / 2.0));
    rect.g = std::max(0.0, std::min(1.0, (gs.f_dc_1 + 1.0) / 2.0));
    rect.b = std::max(0.0, std::min(1.0, (gs.f_dc_2 + 1.0) / 2.0));
    rect.opacity = 1.0 / (1.0 + std::exp(-gs.opacity)); // sigmoid
    
    return rect;
}

// Function to add rectangle to mesh
void add_rectangle_to_mesh(Mesh& mesh, const Rectangle3D& rect) {
    // Calculate the four corner points of the rectangle
    Vector_3 half_width_vec = (rect.width / 2.0) * rect.axis1;
    Vector_3 half_height_vec = (rect.height / 2.0) * rect.axis2;
    
    Point_3 p1 = rect.center + half_width_vec + half_height_vec;
    Point_3 p2 = rect.center - half_width_vec + half_height_vec;
    Point_3 p3 = rect.center - half_width_vec - half_height_vec;
    Point_3 p4 = rect.center + half_width_vec - half_height_vec;
    
    // Add vertices to mesh
    auto v1 = mesh.add_vertex(p1);
    auto v2 = mesh.add_vertex(p2);
    auto v3 = mesh.add_vertex(p3);
    auto v4 = mesh.add_vertex(p4);
    
    // Add two triangular faces to form the rectangle
    mesh.add_face(v1, v2, v3);
    mesh.add_face(v1, v3, v4);
}

// Function to write rectangles to PLY file
void write_ply(const std::vector<Rectangle3D>& rectangles, const std::string& filename) {
    Mesh mesh;
    
    // Convert all rectangles to mesh faces
    for (const auto& rect : rectangles) {
        add_rectangle_to_mesh(mesh, rect);
    }
    
    // Write mesh to PLY file
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot create output file: " + filename);
    }
    
    CGAL::IO::write_PLY(out, mesh);
    out.close();
    
    std::cout << "Written " << rectangles.size() << " rectangles (" 
              << mesh.number_of_faces() << " faces, " 
              << mesh.number_of_vertices() << " vertices) to " << filename << "\n";
}

// Alternative: Write rectangles as simple point cloud with additional properties
void write_ply_simple(const std::vector<Rectangle3D>& rectangles, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot create output file: " + filename);
    }
    
    // Write PLY header
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << rectangles.size() * 4 << "\n";  // 4 corners per rectangle
    out << "property double x\n";
    out << "property double y\n";
    out << "property double z\n";
    out << "property double nx\n";  // Normal (axis1 x axis2)
    out << "property double ny\n";
    out << "property double nz\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "property double width\n";   // Custom properties
    out << "property double height\n";
    out << "property double opacity\n";
    out << "end_header\n";
    
    // Write rectangle data
    for (const auto& rect : rectangles) {
        // Calculate normal as cross product of the two axes
        Vector_3 normal = CGAL::cross_product(rect.axis1, rect.axis2);
        double norm_len = std::sqrt(normal.squared_length());
        if (norm_len > 1e-10) {
            normal = normal / norm_len;
        }
        
        // Calculate corner points
        Vector_3 half_width_vec = (rect.width / 2.0) * rect.axis1;
        Vector_3 half_height_vec = (rect.height / 2.0) * rect.axis2;
        
        std::array<Point_3, 4> corners = {
            rect.center + half_width_vec + half_height_vec,
            rect.center - half_width_vec + half_height_vec,
            rect.center - half_width_vec - half_height_vec,
            rect.center + half_width_vec - half_height_vec
        };
        
        // Convert colors to 0-255 range
        int r = static_cast<int>(rect.r * 255);
        int g = static_cast<int>(rect.g * 255);
        int b = static_cast<int>(rect.b * 255);
        
        // Write each corner
        for (const auto& corner : corners) {
            out << corner.x() << " " << corner.y() << " " << corner.z() << " "
                << normal.x() << " " << normal.y() << " " << normal.z() << " "
                << r << " " << g << " " << b << " "
                << rect.width << " " << rect.height << " " << rect.opacity << "\n";
        }
    }
    
    out.close();
    std::cout << "Written " << rectangles.size() << " rectangles as point cloud to " << filename << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply>\n";
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    try {
        // Step 1: Read Gaussian splats from PLY file
        std::cout << "Reading Gaussian splats from " << input_file << "...\n";
        std::vector<GaussianSplat> gaussians = read_ply(input_file);
        
        // Step 2: Convert Gaussians to rectangles
        std::cout << "Converting Gaussians to 3D rectangles...\n";
        std::vector<Rectangle3D> rectangles;
        rectangles.reserve(gaussians.size());
        
        for (const auto& gs : gaussians) {
            rectangles.push_back(gs_to_rect(gs));
        }
        
        // Step 3: Write rectangles to PLY file
        std::cout << "Writing rectangles to " << output_file << "...\n";
        
        // Choose output format
        if (output_file.find("_simple") != std::string::npos) {
            write_ply_simple(rectangles, output_file);
        } else {
            write_ply(rectangles, output_file);
        }
        
        std::cout << "Conversion completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}