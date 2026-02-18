#ifndef CP_D0_DIST_CPP_H
#define CP_D0_DIST_CPP_H

#include <vector>
#include <cstdint>
#include "cp_d0_dist.hpp"

// Use the same type definitions as the original
#if defined _OPENMP && _OPENMP < 200805
    typedef int32_t index_t;
    #ifndef COMP_T_ON_32_BITS
        typedef int16_t comp_t;
    #else
        typedef int32_t comp_t;
    #endif
#else
    typedef uint32_t index_t;
    #ifndef COMP_T_ON_32_BITS
        typedef uint16_t comp_t;
    #else
        typedef uint32_t comp_t;
    #endif
#endif

// Result structure to hold all outputs
template<typename real_t>
struct CutPursuitResult {
    std::vector<comp_t> components;           // Component assignment for each vertex
    std::vector<real_t> reduced_values;       // Reduced values (D x rV)
    std::vector<std::vector<index_t>> component_lists; // List of vertices in each component (optional)
    
    // Reduced graph structure (optional)
    std::vector<index_t> reduced_first_edge;
    std::vector<comp_t> reduced_adj_vertices;
    std::vector<real_t> reduced_edge_weights;
    
    // Monitoring arrays (optional)
    std::vector<real_t> objectives;
    std::vector<double> times;
    std::vector<real_t> differences;
    
    // Metadata
    comp_t num_components;
    int iterations;
    size_t D; // Dimension
    index_t V; // Number of vertices
};

// Main C++ function
template<typename real_t>
CutPursuitResult<real_t> cp_d0_dist_cpp(
    real_t loss,
    const std::vector<std::vector<real_t>>& Y,  // D x V matrix
    const std::vector<index_t>& first_edge,     // Forward-star representation
    const std::vector<index_t>& adj_vertices,   // Adjacent vertices
    const std::vector<real_t>& edge_weights,    // Edge weights
    const std::vector<real_t>& vert_weights = {},  // Vertex weights (optional)
    const std::vector<real_t>& coor_weights = {},  // Coordinate weights (optional)
    real_t cp_dif_tol = 1e-4,
    int cp_it_max = 10,
    int K = 10,
    int split_iter_num = 50,
    real_t split_damp_ratio = 0.0,
    int kmpp_init_num = 100,
    int kmpp_iter_num = 10,
    real_t min_comp_weight = 0.0,
    int verbose = 1,
    int max_num_threads = 0,
    index_t max_split_size = 1000000,
    bool balance_parallel_split = true,
    bool compute_component_lists = false,
    bool compute_reduced_graph = false,
    bool compute_monitoring = false
) {
    CutPursuitResult<real_t> result;
    
    // Set up dimensions
    size_t D = Y.size();
    index_t V = D > 0 ? Y[0].size() : 0;
    index_t E = adj_vertices.size();
    
    result.D = D;
    result.V = V;
    
    // Flatten Y matrix (column-major order as expected by cp_d0_dist)
    std::vector<real_t> Y_flat(D * V);
    for (size_t d = 0; d < D; ++d) {
        for (index_t v = 0; v < V; ++v) {
            Y_flat[d * V + v] = Y[d][v];
        }
    }
    
    // Prepare edge weights
    const real_t* edge_weights_ptr = edge_weights.empty() ? nullptr : edge_weights.data();
    real_t homo_edge_weight = edge_weights.size() == 1 ? edge_weights[0] : 1.0;
    if (edge_weights.size() <= 1) { edge_weights_ptr = nullptr; }
    
    // Prepare vertex and coordinate weights
    const real_t* vert_weights_ptr = vert_weights.empty() ? nullptr : vert_weights.data();
    const real_t* coor_weights_ptr = coor_weights.empty() ? nullptr : coor_weights.data();
    
    // Set number of threads
    if (max_num_threads <= 0) { 
        #ifdef _OPENMP
        max_num_threads = omp_get_max_threads();
        #else
        max_num_threads = 1;
        #endif
    }
    
    // Prepare monitoring arrays
    std::vector<real_t> objectives, differences;
    std::vector<double> times;
    real_t* obj_ptr = nullptr;
    double* time_ptr = nullptr;
    real_t* dif_ptr = nullptr;
    
    if (compute_monitoring) {
        objectives.resize(cp_it_max + 1);
        times.resize(cp_it_max + 1);
        differences.resize(cp_it_max);
        obj_ptr = objectives.data();
        time_ptr = times.data();
        dif_ptr = differences.data();
    }
    
    // Initialize component array
    result.components.resize(V);
    
    // Create and configure cut-pursuit object
    auto cp = new Cp_d0_dist<real_t, index_t, comp_t>(
        V, E, first_edge.data(), adj_vertices.data(), Y_flat.data(), D
    );
    
    cp->set_loss(loss, Y_flat.data(), vert_weights_ptr, coor_weights_ptr);
    cp->set_edge_weights(edge_weights_ptr, homo_edge_weight);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(max_split_size, K, split_iter_num, split_damp_ratio,
                       kmpp_init_num, kmpp_iter_num);
    cp->set_min_comp_weight(min_comp_weight);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(obj_ptr, time_ptr, dif_ptr);
    cp->set_components(0, result.components.data());
    
    // Run cut-pursuit
    int cp_it = cp->cut_pursuit();
    result.iterations = cp_it;
    
    // Get results
    const index_t* first_vertex;
    const index_t* comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);
    result.num_components = rV;
    
    // Copy reduced values
    const real_t* cp_rX = cp->get_reduced_values();
    result.reduced_values.assign(cp_rX, cp_rX + (D * rV));
    
    // Get component lists if requested
    if (compute_component_lists) {
        result.component_lists.resize(rV);
        for (comp_t rv = 0; rv < rV; ++rv) {
            index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
            result.component_lists[rv].assign(
                comp_list + first_vertex[rv],
                comp_list + first_vertex[rv + 1]
            );
        }
    }
    
    // Get reduced graph if requested
    if (compute_reduced_graph) {
        const comp_t* reduced_edge_list;
        const real_t* reduced_edge_weights_data;
        size_t rE = cp->get_reduced_graph(&reduced_edge_list, &reduced_edge_weights_data);
        
        // Convert to forward-star representation
        result.reduced_first_edge.resize(rV + 1);
        result.reduced_adj_vertices.resize(rE);
        result.reduced_edge_weights.resize(rE);
        
        comp_t rv = 0;
        size_t re = 0;
        while (re < rE || rv < rV) {
            result.reduced_first_edge[rv] = re;
            while (re < rE && reduced_edge_list[2 * re] == rv) {
                result.reduced_adj_vertices[re] = reduced_edge_list[2 * re + 1];
                result.reduced_edge_weights[re] = reduced_edge_weights_data[re];
                re++;
            }
            rv++;
        }
        result.reduced_first_edge[rV] = rE;
    }
    
    // Copy monitoring arrays if requested
    if (compute_monitoring) {
        result.objectives.resize(cp_it + 1);
        result.times.resize(cp_it + 1);
        result.differences.resize(cp_it);
        
        std::copy(objectives.begin(), objectives.begin() + cp_it + 1, result.objectives.begin());
        std::copy(times.begin(), times.begin() + cp_it + 1, result.times.begin());
        std::copy(differences.begin(), differences.begin() + cp_it, result.differences.begin());
    }
    
    // Clean up
    cp->set_components(0, nullptr); // Prevent components from being freed
    delete cp;
    
    return result;
}

// Convenience functions for common use cases
template<typename real_t>
CutPursuitResult<real_t> cut_pursuit_3d_points(
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<index_t>& first_edge,
    const std::vector<index_t>& adj_vertices,
    const std::vector<real_t>& edge_weights = {},
    real_t loss = 1.0, // quadratic loss
    real_t cp_dif_tol = 1e-4,
    int cp_it_max = 10
) {
    // Convert points to D x V format
    std::vector<std::vector<real_t>> Y(3);
    for (int d = 0; d < 3; ++d) {
        Y[d].resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            Y[d][i] = points[i][d];
        }
    }
    
    return cp_d0_dist_cpp<real_t>(loss, Y, first_edge, adj_vertices, edge_weights,
                                 {}, {}, cp_dif_tol, cp_it_max);
}

#endif // CP_D0_DIST_CPP_H