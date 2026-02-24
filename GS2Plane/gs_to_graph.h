#ifndef GS_TO_GRAPH_H
#define GS_TO_GRAPH_H

#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Kernel_traits.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <vector>
#include <memory>
#include <numeric>
#include "custom_ds.h"

namespace Custom {

// 1. Safe Property Map
struct Center_property_map {
  const std::vector<Rectangle3D>* m_rects;
  
  using key_type = std::size_t;
  using value_type = Point_3;
  using reference = Point_3; 
  using category = boost::readable_property_map_tag;
  
  Center_property_map(const std::vector<Rectangle3D>* rects = nullptr) : m_rects(rects) {}
  
  friend Point_3 get(const Center_property_map& map, std::size_t idx) {
    return (*map.m_rects)[idx].center;
  }
};

// 2. Setup Kd-Tree Traits
using Kernel = CGAL::Kernel_traits<Point_3>::Kernel;
using Point_2 = CGAL::Point_2<Kernel>;
using Polygon_2 = CGAL::Polygon_2<Kernel>;
using Traits_base = CGAL::Search_traits_3<Kernel>;
using Tree_Traits = CGAL::Search_traits_adapter<std::size_t, Center_property_map, Traits_base>;
using Kd_Tree = CGAL::Kd_tree<Tree_Traits>;
using Fuzzy_sphere = CGAL::Fuzzy_sphere<Tree_Traits>;
using K_neighbor_search = CGAL::Orthogonal_k_neighbor_search<Tree_Traits>;

class Rectangle_neighbor_query {
public:
  using Item = std::size_t;
private:
  const std::vector<Rectangle3D>& m_rectangles;
  double m_radius;
  double m_h_offset;
  double m_v_offset;
  double m_angle_threshold;
  std::shared_ptr<Kd_Tree> m_tree;
  Center_property_map m_prop_map;
  
public:
  Rectangle_neighbor_query(
    const std::vector<Rectangle3D>& rectangles, double radius = 0.2, 
    double h_offset = 0.10, double v_offset = 0.01, double angle_threshold = 0.9
  )
    : m_rectangles(rectangles), m_radius(radius), 
    m_h_offset(h_offset), m_v_offset(v_offset), m_prop_map(&rectangles), m_angle_threshold(angle_threshold) {
    
    std::vector<std::size_t> indices;
    indices.reserve(rectangles.size());
    for(std::size_t i = 0; i < rectangles.size(); ++i) {
      indices.push_back(i);
    }
    
    std::cout << "Building Kd-tree for " << indices.size() << " rectangles..." << std::endl;
    
    // FIX 1: Explicitly pass iterators, the default splitter, and the traits
    Kd_Tree::Splitter splitter;
    m_tree = std::make_shared<Kd_Tree>(
      indices.begin(), 
      indices.end(), 
      splitter, 
      Tree_Traits(m_prop_map)
    );
    
    std::cout << "Kd-tree built." << std::endl;
  }
  
  void operator()(const Item& query_item, std::vector<Item>& neighbors) const {
    neighbors.clear();
    const auto& query_rect = m_rectangles[query_item];
    // Pre-calculation
    const Point_3& q_center = query_rect.center;
    const Vector_3& q_axis1 = query_rect.axis1;
    const Vector_3& q_axis2 = query_rect.axis2;
    const Vector_3& q_normal = query_rect.normal;
    double limit_u = (query_rect.height / 2.0) + m_h_offset;
    double limit_v = (query_rect.width / 2.0) + m_h_offset;
    double limit_w = m_v_offset; 
    
    double search_radius = m_radius + query_rect.pseudo_radius;
    Fuzzy_sphere sphere(query_rect.center, search_radius, 0.0, Tree_Traits(m_prop_map));
    
    std::vector<Item> candidates;
    m_tree->search(std::back_inserter(candidates), sphere);
    
    for (const Item& candidate_idx : candidates) {
      if (candidate_idx == query_item) continue;
      const auto& candidate = m_rectangles[candidate_idx];

      // Normal Vector Check
      if (candidate.normal * q_normal < m_angle_threshold) continue;
      
      // Vertical Distance Check (Project candidate center onto query normal)
      Vector_3 diff_vec = candidate.center - q_center;
      double dist_w = std::abs(diff_vec * q_normal);
      if (dist_w > limit_w) continue;
      
      auto get_uv = [&](const Point_3& p) {
        Vector_3 vec = p - q_center;
        return std::make_pair(vec * q_axis1, vec * q_axis2);
      };

      auto uv1 = get_uv(candidate.v1);
      auto uv2 = get_uv(candidate.v2);
      auto uv3 = get_uv(candidate.v3);
      auto uv4 = get_uv(candidate.v4);

      // Simplified AABB bounds calculation and examination
      double min_u = std::min({uv1.first, uv2.first, uv3.first, uv4.first});
      double max_u = std::max({uv1.first, uv2.first, uv3.first, uv4.first});
      if (max_u < -limit_u || min_u > limit_u) continue;

      double min_v = std::min({uv1.second, uv2.second, uv3.second, uv4.second});
      double max_v = std::max({uv1.second, uv2.second, uv3.second, uv4.second});
      if (max_v < -limit_v || min_v > limit_v) continue;

      neighbors.push_back(candidate_idx);
    }
  }
};

// Helper for K-NN to avoid adapter issues
struct Point_with_index {
  Point_3 point;
  std::size_t index;

  // Comparison for Kd-tree building logic (required by some splitters though usually pmap handles it)
  bool operator==(const Point_with_index& other) const { 
    return point == other.point && index == other.index; 
  }
};

// Map for Point_with_index
struct Point_with_index_map {
  using key_type = Point_with_index;
  using value_type = Point_3;
  using reference = const Point_3&;
  using category = boost::readable_property_map_tag;

  friend const Point_3& get(const Point_with_index_map&, const Point_with_index& item) {
    return item.point;
  }
};

using Knn_Traits_base = CGAL::Search_traits_3<Kernel>;
using Knn_Traits = CGAL::Search_traits_adapter<Point_with_index, Point_with_index_map, Knn_Traits_base>;
using Knn_Tree = CGAL::Kd_tree<Knn_Traits>;
using Knn_Search = CGAL::Orthogonal_k_neighbor_search<Knn_Traits>;

class KNN_query {
public:
  using Item = std::size_t;

private:
  const std::vector<Rectangle3D>& m_rectangles;
  unsigned int m_k;
  std::shared_ptr<Knn_Tree> m_tree;

public:
  KNN_query(const std::vector<Rectangle3D>& rectangles, unsigned int k = 10)
    : m_rectangles(rectangles), m_k(k) 
  {
    std::cout << "Building K-NN Tree for " << rectangles.size() << " items..." << std::endl;
    
    // Convert input to points with indices
    std::vector<Point_with_index> points;
    points.reserve(rectangles.size());
    for(size_t i = 0; i < rectangles.size(); ++i) {
      points.push_back({rectangles[i].center, i});
    }

    Point_with_index_map pmap;
    Knn_Tree::Splitter splitter;
    m_tree = std::make_shared<Knn_Tree>(
      points.begin(), points.end(), splitter, Knn_Traits(pmap)
    );
    std::cout << "K-NN Tree built." << std::endl;
  }

  // Returns pairs of (Index, Squared Distance)
  void operator()(const Item& query_item, std::vector<std::pair<Item, double>>& neighbors) const {
    neighbors.clear();
    const Point_3& query_point = m_rectangles[query_item].center;

    // Search K+1 because the query point itself will be found with dist 0
    Knn_Search search(*m_tree, query_point, m_k + 1);

    for(auto it = search.begin(); it != search.end(); ++it) {
      // it->first is Point_with_index (because that's what we stored!)
      std::size_t found_idx = it->first.index;
      double sq_dist = it->second;

      if (found_idx != query_item) {
        neighbors.push_back({found_idx, sq_dist});
      }
    }
  }
};

} // namespace Custom

using Rectangle_neighbor_query = Custom::Rectangle_neighbor_query;

// Structure to hold the forward-star graph
struct ForwardStarGraph {
  std::vector<uint32_t> first_edge;    // Size V+1
  std::vector<uint32_t> adj_vertices;  // Size E
  std::vector<double> distances;       // Renamed from edge_weights
};

inline ForwardStarGraph create_forward_star(
  const std::vector<Rectangle3D>& rectangles,
  double neighbor_radius = 0.2,
  double h_offset = 0.1,
  double v_offset = 0.01,
  double angle_threshold = 0.9
) {
  std::cout << "\n==> create_forward_star() called" << std::endl;
  std::cout << "Input: " << rectangles.size() << " rectangles" << std::endl;
  
  if (rectangles.empty()) return {};
  
  // 1. Initialize Neighbor Query Engine
  // Note: increasing radius slightly ensures graph connectivity
  Rectangle_neighbor_query neighbor_query(rectangles, neighbor_radius, h_offset, v_offset, angle_threshold);
  
  std::vector<uint32_t> first_edge;
  std::vector<uint32_t> adj_vertices;
  std::vector<double> distances;
  
  first_edge.reserve(rectangles.size() + 1);
  first_edge.push_back(0); // Start at 0
  
  std::cout << "Building graph..." << std::endl;
  
  size_t total_edges = 0;
  std::vector<std::size_t> current_neighbors;
  
  // 2. Iterate through every rectangle to find its neighbors
  for (std::size_t i = 0; i < rectangles.size(); ++i) {
    if (i % 10000 == 0) std::cout << "\rProcessing node " << i << "..." << std::flush;

    neighbor_query(i, current_neighbors);
    const auto& rect_i = rectangles[i];

    // Add edges to adjacency list
    for (std::size_t neighbor_idx : current_neighbors) {
      if (neighbor_idx == i) continue; 

      const auto& rect_j = rectangles[neighbor_idx];

      // Calculate "Real" surface distance
      double center_dist = std::sqrt(CGAL::squared_distance(rect_i.center, rect_j.center));
      double surf_dist = center_dist - rect_i.pseudo_radius - rect_j.pseudo_radius;
      if (surf_dist < 0.0) surf_dist = 0.0;

      adj_vertices.push_back(static_cast<uint32_t>(neighbor_idx));
      
      // Store raw distance instead of weight
      distances.push_back(surf_dist);
      total_edges++;
    }
    
    first_edge.push_back(static_cast<uint32_t>(total_edges));
  }
  std::cout << std::endl;
  std::cout << "Graph built: " << rectangles.size() << " vertices, " << total_edges << " edges." << std::endl;
  
  return {first_edge, adj_vertices, distances};
}

inline ForwardStarGraph create_forward_star_knn(
  const std::vector<Rectangle3D>& rectangles,
  unsigned int K = 10
) {
  std::cout << "\n==> create_forward_star_knn() called with K=" << K << std::endl;
  
  if (rectangles.empty()) return {};

  // 1. Initialize K-NN Query Engine
  Custom::KNN_query knn_query(rectangles, K);

  ForwardStarGraph graph;
  graph.first_edge.reserve(rectangles.size() + 1);
  // Estimate size: exactly K neighbors per node usually
  graph.adj_vertices.reserve(rectangles.size() * K);
  graph.distances.reserve(rectangles.size() * K);
  
  graph.first_edge.push_back(0); 

  size_t total_edges = 0;
  std::vector<std::pair<std::size_t, double>> current_neighbors;

  // 2. Iterate through every rectangle
  for (std::size_t i = 0; i < rectangles.size(); ++i) {
    if (i % 10000 == 0) std::cout << "\rProcessing node " << i << "..." << std::flush;
    
    // Perform K-NN Search
    knn_query(i, current_neighbors);
    
    const auto& rect_i = rectangles[i];

    for (const auto& neighbor_pair : current_neighbors) {
      std::size_t neighbor_idx = neighbor_pair.first;
      double sq_center_dist = neighbor_pair.second;

      // Calculate Surface Distance
      // KNN returns squared center distance, so sqrt it first
      double center_dist = std::sqrt(sq_center_dist);
      double surf_dist = center_dist - rect_i.pseudo_radius - rectangles[neighbor_idx].pseudo_radius;
      if (surf_dist < 0.0) surf_dist = 0.0;

      graph.adj_vertices.push_back(static_cast<uint32_t>(neighbor_idx));
      graph.distances.push_back(surf_dist);
      total_edges++;
    }
    
    graph.first_edge.push_back(static_cast<uint32_t>(total_edges));
  }

  std::cout << std::endl;
  std::cout << "KNN Graph built: " << rectangles.size() << " vertices, " << total_edges << " edges." << std::endl;
  return graph;
}

#endif // GS_TO_GRAPH_H