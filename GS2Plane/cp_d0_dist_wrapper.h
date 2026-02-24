#ifndef CP_D0_DIST_CPP_H
#define CP_D0_DIST_CPP_H

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include "cp_d0_dist.hpp"

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

// Result structure to hold all possible outputs (matching the Python tuple)
template<typename real_t>
struct CutPursuitResult {
  std::vector<comp_t> components;           
  std::vector<real_t> reduced_values;       // D x rV
  std::vector<std::vector<index_t>> component_lists; 
  
  // Reduced graph structure (Forward-star representation)
  std::vector<index_t> reduced_first_edge;
  std::vector<comp_t> reduced_adj_vertices;
  std::vector<real_t> reduced_edge_weights;
  
  // Monitoring arrays
  std::vector<real_t> objectives;
  std::vector<double> times;
  std::vector<real_t> differences;
  
  // Metadata
  comp_t num_components = 0;
  int iterations = 0;
  size_t D = 0; 
  index_t V = 0; 
};

// 1:1 Core C++ Wrapper (Mirrors cp_d0_dist_cpy)
template<typename real_t>
CutPursuitResult<real_t> cp_d0_dist_cpp(
  real_t loss,
  const std::vector<real_t>& Y_flat,          // Expected as flat D x V
  size_t D,
  index_t V,
  const std::vector<index_t>& first_edge,
  const std::vector<index_t>& adj_vertices,
  const std::vector<real_t>& edge_weights,
  const std::vector<real_t>& vert_weights,
  const std::vector<real_t>& coor_weights,
  real_t cp_dif_tol,
  int cp_it_max,
  int K,
  int split_iter_num,
  real_t split_damp_ratio,
  int kmpp_init_num,
  int kmpp_iter_num,
  real_t min_comp_weight,
  int verbose,
  int max_num_threads,
  index_t max_split_size,
  int balance_parallel_split,
  int compute_List,
  int compute_Graph,
  int compute_Obj,
  int compute_Time,
  int compute_Dif
) {
  CutPursuitResult<real_t> result;
  result.D = D;
  result.V = V;

  // Graph structure
  index_t E = adj_vertices.size();
  
  // Prepare pointers
  const real_t* Y = Y_flat.data();
  const real_t* vert_weights_ptr = vert_weights.empty() ? nullptr : vert_weights.data();
  const real_t* coor_weights_ptr = coor_weights.empty() ? nullptr : coor_weights.data();
  
  const real_t* edge_weights_ptr = edge_weights.data();
  real_t homo_edge_weight = edge_weights.size() == 1 ? edge_weights[0] : 1.0;
  if (edge_weights.size() <= 1) { edge_weights_ptr = nullptr; }

  if (max_num_threads <= 0) { 
    #ifdef _OPENMP
    max_num_threads = omp_get_max_threads();
    #else
    max_num_threads = 1;
    #endif
  }

  // Preallocate Components Array
  result.components.resize(V);
  comp_t* Comp = result.components.data();

  // Preallocate Monitoring Arrays
  if (compute_Obj)  result.objectives.resize(cp_it_max + 1);
  if (compute_Time) result.times.resize(cp_it_max + 1);
  if (compute_Dif)  result.differences.resize(cp_it_max);

  real_t* Obj = compute_Obj ? result.objectives.data() : nullptr;
  double* Time = compute_Time ? result.times.data() : nullptr;
  real_t* Dif = compute_Dif ? result.differences.data() : nullptr;

  // Cut-pursuit with preconditioned forward-Douglas-Rachford
  auto cp = new Cp_d0_dist<real_t, index_t, comp_t>(
    V, E, first_edge.data(), adj_vertices.data(), Y, D
  );

  cp->set_loss(loss, Y, vert_weights_ptr, coor_weights_ptr);
  cp->set_edge_weights(edge_weights_ptr, homo_edge_weight);
  cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
  cp->set_split_param(max_split_size, K, split_iter_num, split_damp_ratio,
                      kmpp_init_num, kmpp_iter_num);
  cp->set_min_comp_weight(min_comp_weight);
  cp->set_parallel_param(max_num_threads, balance_parallel_split);
  cp->set_monitoring_arrays(Obj, Time, Dif);

  // Bind the preallocated component array
  cp->set_components(0, Comp); 

  result.iterations = cp->cut_pursuit();

  // Get number of components and lists
  const index_t* first_vertex;
  const index_t* comp_list;
  comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);
  result.num_components = rV;

  if (compute_List) {
    result.component_lists.resize(rV);
    for (comp_t rv = 0; rv < rV; ++rv) {
      index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
      result.component_lists[rv].assign(
        comp_list + first_vertex[rv],
        comp_list + first_vertex[rv + 1]
      );
    }
  }

  // Copy reduced values (D x rV array)
  const real_t* cp_rX = cp->get_reduced_values();
  result.reduced_values.assign(cp_rX, cp_rX + (D * rV));

  // Retrieve reduced graph structure
  if (compute_Graph) {
    const comp_t* reduced_edge_list;
    const real_t* reduced_edge_weights_data;
    size_t rE = cp->get_reduced_graph(&reduced_edge_list, &reduced_edge_weights_data);

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

  // Resize monitoring arrays to actual iterations used
  if (compute_Obj)  result.objectives.resize(result.iterations + 1);
  if (compute_Time) result.times.resize(result.iterations + 1);
  if (compute_Dif)  result.differences.resize(result.iterations);

  // Prevent internal free() of components array
  cp->set_components(0, nullptr); 
  delete cp;

  return result;
}

// --- Configuration Struct ---
template<typename real_t>
struct CutPursuitParams {
  real_t loss = 1.0;                  
  std::vector<real_t> edge_weights = {};
  std::vector<real_t> vert_weights = {};
  std::vector<real_t> coor_weights = {};
  real_t cp_dif_tol = 1e-4;           
  int cp_it_max = 10;
  int K = 2;                          
  int split_iter_num = 2;            
  real_t split_damp_ratio = 0.0;
  int kmpp_init_num = 3;              
  int kmpp_iter_num = 3;              
  real_t min_comp_weight = 0.0;
  int verbose = 1;
  int max_num_threads = 0;
  index_t max_split_size = 0;
  int balance_parallel_split = 1;
  int compute_List = 0;
  int compute_Graph = 0;
  int compute_Obj = 0;
  int compute_Time = 0;
  int compute_Dif = 0;
};

// Data Marshal & API Wrapper
template<typename real_t, size_t D>
CutPursuitResult<real_t> cut_pursuit_points(
  const std::vector<std::array<real_t, D>>& points,
  const std::vector<index_t>& first_edge,
  const std::vector<index_t>& adj_vertices,
  const CutPursuitParams<real_t>& params = CutPursuitParams<real_t>()
) {
  index_t V = points.size();
  index_t actual_max_split_size = params.max_split_size == 0 ? V : params.max_split_size;
  
  // Transpose NxD (points) to DxN flat array expected by cp_d0_dist.
  std::vector<real_t> Y_flat(D * V);
  for (index_t v = 0; v < V; ++v) {
    for (size_t d = 0; d < D; ++d) {
      Y_flat[v * D + d] = points[v][d];
    }
  }
  
  // Call the core 1:1 C++ wrapper
  return cp_d0_dist_cpp<real_t>(
    params.loss,
    Y_flat,
    D,
    V,
    first_edge, 
    adj_vertices, 
    params.edge_weights,
    params.vert_weights,
    params.coor_weights,
    params.cp_dif_tol, 
    params.cp_it_max,
    params.K, 
    params.split_iter_num,
    params.split_damp_ratio,
    params.kmpp_init_num,
    params.kmpp_iter_num,
    params.min_comp_weight,
    params.verbose,
    params.max_num_threads,
    params.max_split_size,
    params.balance_parallel_split,
    params.compute_List,
    params.compute_Graph,
    params.compute_Obj,
    params.compute_Time,
    params.compute_Dif
  );
}

#endif // CP_D0_DIST_CPP_H