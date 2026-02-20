#ifndef CUSTOM_REGION_GROWING_H
#define CUSTOM_REGION_GROWING_H

#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Kernel_traits.h>
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
using Traits_base = CGAL::Search_traits_3<Kernel>;
using Tree_Traits = CGAL::Search_traits_adapter<std::size_t, Center_property_map, Traits_base>;
using Kd_Tree = CGAL::Kd_tree<Tree_Traits>;
using Fuzzy_sphere = CGAL::Fuzzy_sphere<Tree_Traits>;

class Rectangle_neighbor_query {
public:
  using Item = std::size_t;
private:
  const std::vector<Rectangle3D>& m_rectangles;
  double m_radius;
  double m_radius_offset;
  std::shared_ptr<Kd_Tree> m_tree;
  Center_property_map m_prop_map;
  
public:
  Rectangle_neighbor_query(const std::vector<Rectangle3D>& rectangles, double radius = 0.5)
    : m_rectangles(rectangles), m_radius(radius), m_radius_offset(0.0), m_prop_map(&rectangles) {
    
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
    
    double search_radius = m_radius + query_rect.pseudo_radius;
    Fuzzy_sphere sphere(query_rect.center, search_radius, m_radius_offset, Tree_Traits(m_prop_map));
    
    std::vector<Item> candidates;
    m_tree->search(std::back_inserter(candidates), sphere);
    
    for (const Item& candidate_idx : candidates) {
      if (candidate_idx == query_item) continue;
      const auto& candidate = m_rectangles[candidate_idx];
      
      double normal_dot = std::abs(query_rect.normal * candidate.normal);
      if (normal_dot < 0.5) continue;
      
      double max_possible_radius = m_radius + query_rect.pseudo_radius + candidate.pseudo_radius;
      double distance = CGAL::sqrt(CGAL::squared_distance(query_rect.center, candidate.center));
      
      if (distance <= max_possible_radius) {
        neighbors.push_back(candidate_idx);
      }
    }
  }
};

class Rectangle_region_type {
public:
  using Item = std::size_t;
  using Primitive = Plane_3;

private:
  const std::vector<Rectangle3D>& m_rectangles;
  double m_angle_threshold;
  double m_distance_threshold;
  double m_color_threshold;

  mutable Plane_3 m_current_plane;
  std::array<double, 3> m_current_color;
  bool m_is_valid = false;
  
public:
  Rectangle_region_type(const std::vector<Rectangle3D>& rectangles,
                        double angle_threshold = 0.9,
                        double distance_threshold = 0.5,
                        double color_threshold = 0.2)
    : m_rectangles(rectangles), 
      m_angle_threshold(angle_threshold),
      m_distance_threshold(distance_threshold),
      m_color_threshold(color_threshold) {}
  
  bool is_valid_region(const std::vector<Item>&) const { return m_is_valid; }
  
  bool is_seed(const Item&) const { return true; }
  
  bool update(const std::vector<Item>& region) {
    if (region.empty()) {
      m_is_valid = false;
      return false;
    }

    m_current_plane = Plane_3(m_rectangles[region[0]].center, m_rectangles[region[0]].normal);

    // if (region.size() >= 10) {
    //   std::vector<Point_3> points;
    //   points.reserve(region.size());
    //   for (const auto& idx : region) {
    //     points.push_back(m_rectangles[idx].center);
    //     points.push_back(m_rectangles[idx].v1);
    //     points.push_back(m_rectangles[idx].v2);
    //     points.push_back(m_rectangles[idx].v3);
    //     points.push_back(m_rectangles[idx].v4);
    //   }
      
    //   Plane_3 fitted_plane;
    //   Point_3 centroid;
    //   double quality = CGAL::linear_least_squares_fitting_3(
    //     points.begin(), points.end(), fitted_plane, centroid, 
    //     CGAL::Dimension_tag<0>() 
    //   );
      
    //   if (quality >= 0.8) {
    //     m_current_plane = fitted_plane;
    //   } else {
    //     m_current_plane = Plane_3(m_rectangles[region[0]].center, m_rectangles[region[0]].normal);
    //   }
    // }

    // Calculate the average RGB color of the current region
    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
    for (const auto& idx : region) {
      r_sum += m_rectangles[idx].red;
      g_sum += m_rectangles[idx].green;
      b_sum += m_rectangles[idx].blue;
    }
    double inv_size = 1.0 / region.size();
    m_current_color[0] = r_sum * inv_size;
    m_current_color[1] = g_sum * inv_size;
    m_current_color[2] = b_sum * inv_size;

    m_is_valid = true;
    return true;
  }
  
  // FIX 2: Strict 2-argument signature for CGAL 6.1.1
  bool is_part_of_region(const Item& candidate, const std::vector<Item>& region) const {
    if (region.empty()) return false;
    
    const auto& cand_rect = m_rectangles[candidate];
    double dot_product = std::abs(cand_rect.normal * m_current_plane.orthogonal_vector());
    
    if (dot_product < m_angle_threshold) {
      return false;
    }
    
    double distance = CGAL::sqrt(CGAL::squared_distance(cand_rect.center, m_current_plane));
    if (distance > m_distance_threshold) {
      return false;
    }

    // Calculate Euclidean distance in RGB space
    double dr = cand_rect.red - m_current_color[0];
    double dg = cand_rect.green - m_current_color[1];
    double db = cand_rect.blue - m_current_color[2];
    double color_dist = std::sqrt(dr*dr + dg*dg + db*db);
    // The maximum possible distance in [0,1] RGB space is sqrt(3) ~ 1.732.
    if (color_dist > m_color_threshold) {
      return false;
    }

    return true;
  }
  
  Primitive primitive() const { return m_current_plane; }
};

// 3. Custom Region Map
struct Region_map_type {
  std::vector<std::size_t>* m_vec;
  
  using key_type = std::size_t;
  using value_type = std::size_t;
  using reference = std::size_t&;
  using category = boost::read_write_property_map_tag;
  
  Region_map_type(std::vector<std::size_t>& vec) : m_vec(&vec) {}
  
  friend std::size_t get(const Region_map_type& map, std::size_t idx) {
    return (*map.m_vec)[idx];
  }
  friend void put(const Region_map_type& map, std::size_t idx, std::size_t val) {
    (*map.m_vec)[idx] = val;
  }
};

} // namespace Custom

using Rectangle_neighbor_query = Custom::Rectangle_neighbor_query;
using Rectangle_region_type = Custom::Rectangle_region_type;
using Region_map_type = Custom::Region_map_type;

using Rectangle_region_growing = CGAL::Shape_detection::Region_growing<
    Rectangle_neighbor_query, 
    Rectangle_region_type, 
    Region_map_type>;

inline std::vector<std::vector<std::size_t>> detect_planar_regions(
  const std::vector<Rectangle3D>& rectangles,
  double neighbor_radius = 0.2,
  double angle_threshold = 0.9,
  double distance_threshold = 0.5,
  double color_threshold = 0.20,
  std::size_t min_region_size = 10
) {
  std::cout << "\n==> detect_planar_regions() called" << std::endl;
  std::cout << "Input: " << rectangles.size() << " rectangles" << std::endl;
  
  if (rectangles.empty()) return {};
  
  std::vector<std::size_t> items(rectangles.size());
  std::iota(items.begin(), items.end(), 0);

  std::vector<std::size_t> region_map_vec(rectangles.size(), std::size_t(-1));
  Region_map_type region_map(region_map_vec);
  
  Rectangle_neighbor_query neighbor_query(rectangles, neighbor_radius);
  Rectangle_region_type region_type(rectangles, angle_threshold, distance_threshold, color_threshold);
  Rectangle_region_growing region_growing(
    items, 
    neighbor_query, 
    region_type, 
    region_map
  );
  
  std::vector<typename Rectangle_region_growing::Primitive_and_region> regions_with_primitives;
  region_growing.detect(std::back_inserter(regions_with_primitives));
  
  std::vector<std::vector<std::size_t>> regions;
  for (const auto& region_pair : regions_with_primitives) {
    if (region_pair.second.size() >= min_region_size) {
      regions.push_back(region_pair.second);
    }
  }
  
  std::cout << "==> Found " << regions.size() << " planar regions\n";
  return regions;
}

#endif // CUSTOM_REGION_GROWING_H