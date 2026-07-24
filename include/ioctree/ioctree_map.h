#ifndef IOCTREE_MAP_H
#define IOCTREE_MAP_H

#include <Eigen/Core>
#include <map>
#include <shared_mutex>
#include <tuple>
#include <vector>

#include "i_octree.h"

// 八叉树地图的配置参数
struct IOctreeConfig {
  int bucket_size = 32;  // 叶子节点最大点数，超过则触发 Morton 分桶分裂
  int lru_capacity =
      500000;  // LRU 缓存最大容量，超限时自动淘汰最久未修改的叶子
  float min_extent = 0.1;  // 最小八叉树节点半边长，小于此值不再分裂
  float voxel_downsample_size =
      0.3;  // 体素降采样栅格大小，用于新点插入时的稀疏化
  float points_dis_min = 0.01;  // 新点与最近邻点的最小距离阈值，小于该值则丢弃
  bool store_index = false;  // 是否在第4维存储原始点云索引（KNN返回原始索引时需要）
};

// 基于增量八叉树的局部点云地图封装。
// 在 Octree 之上提供体素降采样、线程安全、配置管理等上层功能，
// 用于 SLAM 前端的局部建图和最近邻查询。
template <typename PointT>
class IOctreeMap {
 public:
  using PointVector = std::vector<PointT, Eigen::aligned_allocator<PointT>>;

  // 构造：初始化 Octree 并将配置参数写入底层。
  explicit IOctreeMap(const IOctreeConfig &config)
      : voxel_downsample_size_(config.voxel_downsample_size),
        points_dis_min_(config.points_dis_min) {
    octree_ = new thuni::Octree();
    octree_->SetBucketSize(config.bucket_size);
    octree_->SetMinExtent(config.min_extent);
    octree_->SetLruCapacity(config.lru_capacity);
    octree_->store_index_ = config.store_index;
    octree_->dim_ = config.store_index ? 4 : 3;
  }

  ~IOctreeMap() { delete octree_; }

  // 从点云全量构建八叉树，清空已有数据。
  void Build(const PointVector &point_cloud) {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);
    octree_->Initialize(point_cloud);
  }

  // 增量插入新点到八叉树，同时进行体素降采样和近距离去重。
  // nearest_points 为每个新点在已有地图中的最近邻，用于判断是否需要插入。
  void AddPoints(const PointVector &points_to_add,
                 const std::vector<PointVector> &nearest_points) {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);

    const int ds_point_cloud_size = points_to_add.size();

    // 树为空时直接全量构建
    if (octree_->Size() == 0) {
      if (ds_point_cloud_size > 5) {
        octree_->Initialize(points_to_add);
      }
      return;
    }

    PointVector point_to_add;
    point_to_add.reserve(ds_point_cloud_size);
    PointVector point_no_need_downsample;
    point_no_need_downsample.reserve(ds_point_cloud_size);

    for (int i = 0; i < ds_point_cloud_size; i++) {
      if (!nearest_points[i].empty()) {
        const PointVector &points_near = nearest_points[i];

        // 距离过近：丢弃，不插入
        if (CalcDist(points_to_add[i], points_near[0]) < points_dis_min_) {
          continue;
        }

        // 计算新点所属的体素中心
        PointT mid_point;
        mid_point.x = std::floor(points_to_add[i].x / voxel_downsample_size_) *
                          voxel_downsample_size_ +
                      0.5 * voxel_downsample_size_;
        mid_point.y = std::floor(points_to_add[i].y / voxel_downsample_size_) *
                          voxel_downsample_size_ +
                      0.5 * voxel_downsample_size_;
        mid_point.z = std::floor(points_to_add[i].z / voxel_downsample_size_) *
                          voxel_downsample_size_ +
                      0.5 * voxel_downsample_size_;

        const float dist = CalcDist(points_to_add[i], mid_point);

        // 最近邻不在同一体素且距离较远：该区域无需降采样，直接插入
        if (fabs(points_near[0].x - mid_point.x) >
                0.5 * voxel_downsample_size_ &&
            fabs(points_near[0].y - mid_point.y) >
                0.5 * voxel_downsample_size_ &&
            fabs(points_near[0].z - mid_point.z) >
                0.5 * voxel_downsample_size_) {
          point_no_need_downsample.push_back(points_to_add[i]);
          continue;
        }

        // 体素降采样判断：新点比体素内所有已有最近邻更靠近体素中心才插入
        bool need_add = true;
        const int num_near = static_cast<int>(points_near.size());
        for (int j = 0; j < num_near; j++) {
          if (CalcDist(points_near[j], mid_point) < dist) {
            need_add = false;
            break;
          }
        }
        if (need_add) {
          point_to_add.push_back(points_to_add[i]);
        }
      } else {
        // 无最近邻（新区域）：直接插入
        point_to_add.push_back(points_to_add[i]);
      }
    }

    if (!point_to_add.empty()) {
      octree_->Update(point_to_add);
    }
    if (!point_no_need_downsample.empty()) {
      octree_->Update(point_no_need_downsample);
    }

    // 第二层降采样：体素内只保留最靠近中心的一个点（ikd-Tree 风格）
    // 仅对 point_to_add（同体素竞争插入的）做完整清理，
    // point_no_need_downsample（新区域）直接插入，不做第二层。
    // 与原始 FAST-LIO2 中 ikdtree->Add_Points(PointToAdd, true) 一致。
    if (!point_to_add.empty()) {
      // 按体素坐标分组，避免同一批中同体素多点重复处理
      std::map<std::tuple<int, int, int>, PointVector> voxel_groups;
      for (const auto &pt : point_to_add) {
        int ix = static_cast<int>(std::floor(pt.x / voxel_downsample_size_));
        int iy = static_cast<int>(std::floor(pt.y / voxel_downsample_size_));
        int iz = static_cast<int>(std::floor(pt.z / voxel_downsample_size_));
        voxel_groups[std::make_tuple(ix, iy, iz)].push_back(pt);
      }

      for (auto &kv : voxel_groups) {
        const auto &group = kv.second;
        const auto &first = group[0];

        float box_min[3], box_max[3];
        box_min[0] = std::floor(first.x / voxel_downsample_size_) *
                     voxel_downsample_size_;
        box_min[1] = std::floor(first.y / voxel_downsample_size_) *
                     voxel_downsample_size_;
        box_min[2] = std::floor(first.z / voxel_downsample_size_) *
                     voxel_downsample_size_;
        box_max[0] = box_min[0] + voxel_downsample_size_;
        box_max[1] = box_min[1] + voxel_downsample_size_;
        box_max[2] = box_min[2] + voxel_downsample_size_;

        PointT mid_point;
        mid_point.x = box_min[0] + 0.5f * voxel_downsample_size_;
        mid_point.y = box_min[1] + 0.5f * voxel_downsample_size_;
        mid_point.z = box_min[2] + 0.5f * voxel_downsample_size_;

        // 搜出体素内所有已有旧点
        PointVector voxel_points;
        octree_->SearchByBox(box_min, box_max, voxel_points);

        // 找最靠近体素中心的（已有 + 新点中）
        float min_dist = CalcDist(first, mid_point);
        PointT best = first;
        for (const auto &vp : voxel_points) {
          float d = CalcDist(vp, mid_point);
          if (d < min_dist) {
            min_dist = d;
            best = vp;
          }
        }
        for (size_t gi = 1; gi < group.size(); gi++) {
          float d = CalcDist(group[gi], mid_point);
          if (d < min_dist) {
            min_dist = d;
            best = group[gi];
          }
        }

        // 删掉体素内所有旧点，只插最好的那个
        octree_->BoxWiseDelete(box_min, box_max, true);
        PointVector best_vec = {best};
        octree_->Update(best_vec);
      }
    }
  }

  // KNN 最近邻搜索：返回查询点周围最近的 k 个点及其平方距离。
  void NearestSearch(const PointT &point, int k_nearest,
                     PointVector &nearest_points,
                     std::vector<float> &point_distance) const {
    std::shared_lock<std::shared_mutex> lock(tree_mutex_);
    std::vector<PointT> result_pts;
    std::vector<float> distances;
    octree_->KnnNeighbors(point, k_nearest, result_pts, distances);
    nearest_points.assign(result_pts.begin(), result_pts.end());
    point_distance = std::move(distances);
  }

  // 修改八叉树最小半边长
  void SetMinExtent(float min_extent) { octree_->SetMinExtent(min_extent); }

  // 返回当前地图中的总点数
  int PointsSize() { return static_cast<int>(octree_->Size()); }

  // 用新配置重建 Octree，释放旧树并重新初始化。
  void Reset(const IOctreeConfig &config) {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);
    delete octree_;
    octree_ = new thuni::Octree();
    octree_->SetBucketSize(config.bucket_size);
    octree_->SetMinExtent(config.min_extent);
    octree_->SetLruCapacity(config.lru_capacity);
    octree_->store_index_ = config.store_index;
    octree_->dim_ = config.store_index ? 4 : 3;
    voxel_downsample_size_ = config.voxel_downsample_size;
    points_dis_min_ = config.points_dis_min;
  }

  // 删除指定 AABB 区域内的点（物理删除）。
  void DeleteRegion(const float *min, const float *max) {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);
    octree_->BoxWiseDelete(min, max, true);
  }

 private:
  // 计算两点之间的平方欧氏距离。
  static float CalcDist(const PointT &p1, const PointT &p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx * dx + dy * dy + dz * dz;
  }

  thuni::Octree *octree_ = nullptr;
  mutable std::shared_mutex tree_mutex_;

  float voxel_downsample_size_ = 0.3;
  float points_dis_min_ = 0.01;
};

#endif
