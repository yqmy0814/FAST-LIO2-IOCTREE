#ifndef THUNI_OCTREE_H_
#define THUNI_OCTREE_H_

// Copyright (c) 2023 Jun Zhu, Tsinghua University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace thuni {

class Octree;

// 距离-索引对，用于 KNN 搜索结果的有序数组。
struct DistanceIndex {
  float dist;    // 平方距离
  float *index;  // 指向点数据的指针

  DistanceIndex() { dist = std::numeric_limits<float>::max(); }

  DistanceIndex(float dist, float *index) : dist(dist), index(index) {}

  // 用于插入排序：距离小的排前面
  bool operator<(const DistanceIndex &dist_index) const {
    return dist < dist_index.dist;
  }
};

// KNN 搜索结果集：有序数组实现，维护 k 个最近邻候选点。
class KNNSimpleResultSet {
 private:
  size_t capacity_;                        // k：最大候选数
  size_t count_;                           // 当前候选数
  float worst_distance_;                   // 当前最远距离，用于剪枝
  std::vector<DistanceIndex> dist_index_;  // 有序数组，按距离升序

 public:
  // 构造：预分配 capacity 个槽位，避免运行时 realloc
  KNNSimpleResultSet(size_t capacity_) : capacity_(capacity_) {
    dist_index_.resize(capacity_, DistanceIndex());
    Clear();
  }

  // 返回内部有序数组的只读引用
  const std::vector<DistanceIndex> &GetData() { return dist_index_; }

  ~KNNSimpleResultSet() {}

  // 重置结果集，清空所有候选点
  void Clear() {
    worst_distance_ = std::numeric_limits<float>::max();
    dist_index_[capacity_ - 1].dist = worst_distance_;
    count_ = 0;
  }

  // 当前候选数
  size_t Size() const { return count_; }

  // 候选集是否已满（k 个点都已找到）
  bool Full() const { return count_ == capacity_; }

  // 尝试插入一个候选点。距离 >= worst 的直接丢弃；
  // 否则插入排序到正确位置，更新 worst_distance_。
  void AddPoint(float dist, float *index) {
    if (dist >= worst_distance_) return;

    if (count_ < capacity_) ++count_;
    size_t i;
    for (i = count_ - 1; i > 0; --i) {
      if (dist_index_[i - 1].dist > dist)
        dist_index_[i] = dist_index_[i - 1];
      else
        break;
    }
    dist_index_[i].dist = dist;
    dist_index_[i].index = index;
    worst_distance_ = dist_index_[capacity_ - 1].dist;
  }

  // 返回当前最远候选点的距离，用于 overlaps/Inside 剪枝判断
  float WorstDist() const { return worst_distance_; }
};

// AABB 区域删除的包围盒定义，min/max 为对角点坐标。
struct BoxDeleteType {
  float min[3];  // 包围盒最小角 (x, y, z)
  float max[3];  // 包围盒最大角 (x, y, z)

  // 调试用：打印包围盒坐标
  void Show() {
    printf("min:(%f, %f, %f), max:(%f, %f, %f)\n", min[0], min[1], min[2],
           max[0], max[1], max[2]);
  }
};

// 搜索性能统计：记录深度、访问节点数、耗时等，用于调试和性能分析。
struct RunDetails {
  int depth;      // 搜索深度
  int pts_n;      // 搜索的点数
  int node_n;     // 搜索的叶子节点数
  bool one_path;  // 首次搜索的单条路径
  std::chrono::high_resolution_clock::time_point begin;
  std::chrono::high_resolution_clock::time_point end;
  float total_time;

  RunDetails() { Clear(); }

  // 开始计时
  void Start() { begin = std::chrono::high_resolution_clock::now(); }

  // 结束计时，返回本次耗时（毫秒），并累加到 total_time
  float End() {
    end = std::chrono::high_resolution_clock::now();
    float time =
        float(std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                  .count()) /
        1e3;
    begin = end;
    total_time += time;
    return time;
  }

  // 重置所有统计字段
  void Clear() {
    depth = pts_n = node_n = 0;
    one_path = true;
    total_time = 0.0f;
  }

  // 打印统计信息
  void Show() {
    printf("octree\t depth: %d, node_n: %d, pts_n: %d\n\n", depth, node_n,
           pts_n);
  }

  friend std::ostream &operator<<(std::ostream &os, const RunDetails &c) {
    os << "octree depth: " << c.depth << ", node_n: " << c.node_n
       << ", pts_n: " << c.pts_n << std::endl;
    return os;
  }
};

// 八叉树节点：根据 child 是否为空分为枝干节点和叶子节点。
// 枝干节点：child[8] 指向 8 个子象限，points 为空。
// 叶子节点：child 为空，points 指向连续内存块（new float[size * dim_]），
//           points[i] = 块内第 i 个点的首地址，析构时只需 delete[] points[0]。
// 每个节点持有 lru_idx 和 lru_owner_ 用于 LRU 自管理：
//   - lru_idx >= 0 表示已注册到 LRU，析构时自动移除
//   - lru_owner_ 指向所属 Octree 实例
class Octant {
 public:
  bool is_active_;   // 软删除标记，false 时搜索跳过但不释放
  float x_, y_, z_;  // 节点中心坐标
  float extent_;     // 半边长（节点边长的一半）
  std::vector<float *> points_;  // 叶子节点的点数据指针数组
  Octant **child_;  // 枝干节点的 8 个子节点指针，叶子为 nullptr
  int lru_idx_ = -1;             // LRU 数组下标，-1 = 未注册
  Octree *lru_owner_ = nullptr;  // 所属 Octree 实例，析构时用于调用 LruRemove

  Octant();

  ~Octant();

  // 递归计算以本节点为根的子树中的总点数
  size_t Size();

  // 递归辅助：累加子树中的点数到 size_
  void GetOctantSize(Octant *octant, size_t &size_);

  // 初始化 8 个子节点指针数组（全部置 nullptr）
  void InitChild();
};

// 节点信息记录器：收集八叉树节点引用，用于离线分析和调试输出。
class InfoRecord {
 public:
  std::vector<Octant *> octant_vec_;  // 记录的节点列表

  InfoRecord() {}

  // 添加一个节点到记录列表
  void AddOctant(Octant *octant) {
    if (octant == 0) return;
    octant_vec_.push_back(octant);
  }

  // 将记录的节点信息写入文本文件（中心坐标 + extent）
  void WriteToTxt(std::string name = "otree_info") {
    {
      std::string filename =
          "/media/zhujun/0DFD06D20DFD06D2/SLAM/octree_test/ikd-Tree-main/data/";
      filename += name + ".txt";
      std::ofstream log_file(filename, std::ios::out);
      log_file << "# x y z extent" << std::endl;
      for (int i = 0; i < octant_vec_.size(); i++) {
        log_file << octant_vec_[i]->x_ << " " << octant_vec_[i]->y_ << " "
                 << octant_vec_[i]->z_ << " " << octant_vec_[i]->extent_
                 << "\n";
      }
      log_file.close();
    }
  }

  void Clear() { octant_vec_.clear(); }
};

// 增量八叉树，支持动态插入、区域删除、KNN 和半径搜索。
// 叶子节点的点数据由 new float[size * dim_] 连续分配，减少内存碎片。
// dim_=3 时每点仅存 xyz，dim_=4 时额外存储一维原始索引（设 store_index_=true）。
class Octree {
 public:
  size_t bucket_size_;        // 叶子节点最大点数，超过则分裂
  float min_extent_;          // 最小体素半径，小于此值不再分裂
  bool store_index_ = false;  // 是否在第4维存储原始点索引
  int dim_ = 3;               // 每点维度，由 store_index_ 决定
  // Morton 码优先遍历序表。KNN
  // 搜索时先访问查询点所在的子象限（ordered_indies[code][0]），
  // 再按距离由近到远访问其余7个子象限，配合 Inside() 实现早停剪枝。
  size_t ordered_indices_[8][7] = {
      {1, 2, 4, 3, 5, 6, 7}, {0, 3, 5, 2, 4, 7, 6}, {0, 3, 6, 1, 4, 7, 5},
      {1, 2, 7, 0, 5, 6, 4}, {0, 5, 6, 1, 2, 7, 3}, {1, 4, 7, 0, 3, 6, 2},
      {2, 4, 7, 0, 3, 5, 1}, {3, 5, 6, 1, 2, 4, 0}};
  RunDetails run_details_;
  InfoRecord info_record_;

  Octree();

  ~Octree();

  // === 配置 ===

  // 设置最小分裂半边长，小于此值的叶子不再分裂
  void SetMinExtent(float extent);

  // 设置叶子节点最大点数，超过则分裂
  void SetBucketSize(size_t bucket_size);

  // 设置 LRU 最大容量，超限时自动淘汰最久未修改的叶子
  void SetLruCapacity(int capacity);

  // === LRU 管理 ===

  // 数组模拟双向链表，淘汰最久未修改的叶子。
  // 叶子在四个时机操作 LRU：构造（注册）、修改（touch
  // MRU）、分裂（移除）、析构（移除）。
  struct LruNode {
    Octant *octant;  // 指向八叉树节点
    int prev;        // 链表前驱下标，-1 = 链表头（MRU）
    int next;        // 链表后继下标，-1 = 链表尾（LRU）
  };
  std::vector<LruNode> lru_nodes_;           // 链表节点数组
  std::vector<int> lru_free_slots_;          // 空闲槽位回收列表
  int lru_mru_ = -1, lru_lru_ = -1;          // 链表头尾下标
  int lru_count_ = 0;                        // 当前 LRU 节点数
  int lru_capacity_ = 500000;                // LRU 最大容量

  // 从 LRU 链表中移除指定条目
  void LruRemove(int idx);

  // 分配一个 LRU 链表节点。优先复用空闲槽位，否则追加新元素。
  int LruAllocNode(Octant *octant);

  // 释放一个 LRU 链表节点：槽位回收到空闲列表
  void LruFreeNode(int idx);

  // 将节点移到链表头部（MRU 端）。已在头部时直接返回。
  void LruTouch(int idx);

  // 超容量时循环淘汰 LRU 尾部叶子，直到 lru_count_ <= lru_capacity_。
  // 在 Initialize() 和 Update() 末尾自动调用。
  void LruEvictIfNeeded();

  // === 节点辅助 ===

  // 返回当前树中的有效点数（总插入点数 - 已删除点数）。
  size_t Size();

  // 收集以 octant 为根的子树中所有节点，min_extent > 0 时过滤小节点。
  void GetNodes(Octant *octant, std::vector<Octant *> &nodes,
                float min_extent = 0);

  // 提取树中所有点的坐标到容器中。
  template <typename PointT, typename ContainerT>
  ContainerT GetData();

  // 收集以 octant 为根的子树中所有叶子节点。
  void GetLeafNodes(const Octant *octant, std::vector<const Octant *> &nodes);

  // === 构建 / 更新 ===

  // 从点云构建八叉树，清空已有数据。
  template <typename ContainerT>
  void Initialize(ContainerT &pts_);

  // 增量插入新点。若树为空则调用 initialize；否则扩展根节点包围盒后插入。
  template <typename ContainerT>
  void Update(ContainerT &pts_);

  // 清空八叉树及 LRU 缓存，释放所有内存。
  void Clear();

  // === 删除 ===

  // clear_data=true: 物理删除点数据；false: 仅标记 is_active_=false（软删除）
  void BoxWiseDelete(const BoxDeleteType &box_range, bool clear_data);
  void BoxWiseDelete(const float *min, const float *max, bool clear_data);

  // === 搜索 — 公开接口 ===

  // 返回类型约定：
  //   vector<PointT> — 返回点坐标（不依赖 store_index_）
  //   vector<size_t> / vector<int> — 返回原始点云索引（需 store_index_=true）

  // 半径搜索，返回索引（需 store_index_=true）
  template <typename PointT>
  void RadiusNeighbors(const PointT &query_in, float radius,
                       std::vector<size_t> &result_indices);

  // 半径搜索，返回索引+距离（需 store_index_=true）
  template <typename PointT>
  void RadiusNeighbors(const PointT &query_in, float radius,
                       std::vector<size_t> &result_indices,
                       std::vector<float> &distances);

  // 半径搜索，返回点坐标+距离
  template <typename PointT>
  void RadiusNeighbors(const PointT &query_in, float radius,
                       std::vector<PointT> &result_indices,
                       std::vector<float> &distances);

  // 半径搜索，返回点坐标
  template <typename PointT>
  void RadiusNeighbors(const PointT &query_in, float radius,
                       std::vector<PointT> &result_indices);

  // KNN 搜索，返回点坐标+距离
  template <typename PointT>
  int32_t KnnNeighbors(const PointT &query_in, int k,
                       std::vector<PointT> &result_indices,
                       std::vector<float> &distances);

  // KNN 搜索，返回索引+距离（需 store_index_=true）
  template <typename PointT>
  int32_t KnnNeighbors(const PointT &query_in, int k,
                       std::vector<size_t> &result_indices,
                       std::vector<float> &distances);

  // KNN 搜索，返回索引(int)+距离（需 store_index_=true）
  template <typename PointT>
  int32_t KnnNeighbors(const PointT &query_in, int k,
                       std::vector<int> &result_indices,
                       std::vector<float> &distances);

  // 按 AABB 范围搜索：返回指定轴对齐包围盒范围内的所有点坐标。
  // min/max 为包围盒对角点坐标，result_points 按 PointT 格式返回。
  template <typename ContainerT>
  void SearchByBox(const float *min, const float *max,
                   ContainerT &result_points);

  // === 记录 / 调试 ===

  // 将树中所有节点信息写入文本文件（调试用）。
  void WriteNodeInfoToTxt(float min_extent = 0);

  // 从点云构建八叉树，同时记录节点信息（调试/分析用）。
  template <typename ContainerT>
  void InitializeRecord(ContainerT &points);

  // 创建带 info_record 记录的八叉树节点（调试/分析用）。
  Octant *CreateOctantRecord(float x, float y, float z, float extent,
                             std::vector<float *> &points);

  // 增量插入新点，同时扩展根节点包围盒并记录节点信息（调试/分析用）。
  template <typename ContainerT>
  void UpdateRecord(ContainerT &points);

  // 增量插入新点到子树，同时记录节点信息（调试/分析用）。
  void UpdateOctantRecord(Octant *octant, const std::vector<float *> &points);

  // 将 info_record_ 中的节点信息写入文本文件，可附带清空。
  void WriteInfoRecord(std::string name = "otree_info", bool clear = false);

  void WriteRemainNodeRecord(std::string name = "otree_info");

 protected:
  Octant *root_;                           // 树根节点
  size_t last_pts_num_, pts_num_deleted_;  // 总插入点数、已删除点数

  Octree(Octree &);                      // 禁止拷贝
  Octree &operator=(const Octree &oct);  // 禁止赋值

  // 递归创建八叉树节点
  // 超限则按 Morton 码分桶递归创建 8 个子节点，否则成为叶子并注册 LRU。
  Octant *CreateOctant(float x, float y, float z, float extent,
                       std::vector<float *> &points);

  // 增量插入新点到以 octant 为根的子树
  // 叶子超限时分裂（先从 LRU 移除），否则追加点数据并 touch MRU。
  void UpdateOctant(Octant *octant, const std::vector<float *> &points);

  // 递归删除 AABB 区域内的节点。
  // Case 1: octant 完全在 AABB 内 → 整棵子树删除
  // Case 2: 叶子节点部分在 AABB 内 → 逐点过滤，保留外部点
  // Case 3: 枝干节点部分在 AABB 内 → 递归子节点，子节点全删则级联删除当前节点
  void BoxWiseDelete(Octant *octant, const BoxDeleteType &box_range,
                     bool &deleted, bool clear_data);

  // === 几何测试 ===

  // 判断搜索球 S(q,r) 与 octant 的 AABB 是否有交集。
  // 利用对称性简化为正象限的 Minkowski 和测试。
  bool Overlaps(const float *query, float sqr_radius, const Octant *o);

  // 判断搜索球 S(q,r) 是否完全包住 octant 的 AABB。
  // 利用对称性：取最远角点判断是否在球内。
  bool Contains(const float *query, float sqr_radius, const Octant *o);

  // 判断搜索球 S(q,r) 是否完全在 octant 的 AABB 内部。
  // 用于 KNN 早停：球完全在节点内意味着所有更优解已在球内。
  bool Inside(const float *query, float sqr_radius, const Octant *octant);

  // === 搜索 — 内部实现 ===

  // 半径搜索递归实现，返回点坐标指针。
  void RadiusNeighbors(const Octant *octant, const float *query, float radius,
                       float sqr_radius, std::vector<float *> &result_indices);

  // 半径搜索递归实现，返回点坐标指针和对应距离。
  void RadiusNeighbors(const Octant *octant, const float *query, float radius,
                       float sqr_radius, std::vector<float *> &result_indices,
                       std::vector<float> &distances);

  // 半径搜索递归实现（带索引），搜索球完全在节点内时通过 Inside() 早停。
  bool RadiusNeighborsFast(const Octant *octant, const float *query,
                        float sqr_radius, std::vector<size_t> &result_indices,
                        std::vector<float> &distances);

  // KNN 搜索核心递归：沿 Morton 码优先路径下降，堆满后通过 Inside() 剪枝早停。
  // 返回 true 表示搜索球完全在当前 octant 内，无需继续回溯。
  bool KnnNeighbors(const Octant *octant, const float *query,
                    KNNSimpleResultSet &heap);

  // 按 AABB 范围搜索递归实现：收集指定包围盒内的所有点数据指针。
  void SearchByBox(const Octant *octant, const float *min, const float *max,
                   std::vector<float *> &result_indices);
};

/*----------------------------------------------------------------------------*/
// === 构造 / 析构 ===

// 构造：初始化为原点位置、零半边长、非活跃状态，无子节点。
inline Octant::Octant() : x_(0.0f), y_(0.0f), z_(0.0f), extent_(0.0f) {
  child_ = nullptr;
  is_active_ = true;
}

// 析构函数：枝干递归删除子节点；叶子释放点数据并从 LRU 自动移除。
inline Octant::~Octant() {
  if (child_ != nullptr) {
    for (size_t i = 0; i < 8; ++i) {
      if (child_[i] != 0) delete child_[i];
    }
    delete[] child_;
    child_ = nullptr;
  } else {
    // 叶子析构时自动从 LRU 移除
    if (lru_idx_ >= 0 && lru_owner_) {
      lru_owner_->LruRemove(lru_idx_);
      lru_idx_ = -1;
    }
    delete[] points_[0];
    std::vector<float *>().swap(points_);
  }
}

// 构造：使用默认配置初始化八叉树。
inline Octree::Octree() : bucket_size_(32), min_extent_(0.1f), root_(0) {
  pts_num_deleted_ = last_pts_num_ = 0;
  dim_ = 3;
}

// 析构：清空树和 LRU 缓存。
inline Octree::~Octree() { Clear(); }

// === 配置 ===

// 设置叶子节点最小分裂半边长，小于此值的叶子不再分裂。
inline void Octree::SetMinExtent(float extent) { min_extent_ = extent; }

// 设置叶子节点最大点数，超过则触发 Morton 码分桶分裂。
inline void Octree::SetBucketSize(size_t bucket_size) {
  bucket_size_ = bucket_size;
}

// 设置 LRU 缓存最大容量，超限时在 Initialize()/Update() 末尾自动淘汰。
inline void Octree::SetLruCapacity(int capacity) { lru_capacity_ = capacity; }

// === LRU 管理 ===

// 为 octant 分配 LRU 链表节点。
// 优先复用空闲槽位，否则在数组末尾追加，返回分配的下标。
inline int Octree::LruAllocNode(Octant *octant) {
  int idx;
  if (!lru_free_slots_.empty()) {
    idx = lru_free_slots_.back();
    lru_free_slots_.pop_back();
    lru_nodes_[idx] = {octant, -1, -1};
  } else {
    idx = static_cast<int>(lru_nodes_.size());
    lru_nodes_.push_back({octant, -1, -1});
  }
  lru_count_++;
  return idx;
}

// 释放 LRU 节点占用的槽位，回收到空闲列表供复用。
inline void Octree::LruFreeNode(int idx) {
  lru_nodes_[idx].octant = nullptr;
  lru_free_slots_.push_back(idx);
  lru_count_--;
}

// 将指定 LRU 节点移到链表头部（MRU 端），表示最近被访问。
// 已在头部时直接返回；同时维护 lru_mru_ / lru_lru_ 指针。
inline void Octree::LruTouch(int idx) {
  LruNode &node = lru_nodes_[idx];
  if (idx == lru_mru_) return;
  if (node.prev >= 0) lru_nodes_[node.prev].next = node.next;
  if (node.next >= 0) lru_nodes_[node.next].prev = node.prev;
  if (idx == lru_lru_) lru_lru_ = node.prev;
  node.prev = -1;
  node.next = lru_mru_;
  if (lru_mru_ >= 0) lru_nodes_[lru_mru_].prev = idx;
  lru_mru_ = idx;
  if (lru_lru_ == -1) lru_lru_ = idx;
}

// 从 LRU 链表中移除指定条目，同时更新头尾指针。
inline void Octree::LruRemove(int idx) {
  LruNode &node = lru_nodes_[idx];
  if (node.prev >= 0) lru_nodes_[node.prev].next = node.next;
  if (node.next >= 0) lru_nodes_[node.next].prev = node.prev;
  if (idx == lru_mru_) lru_mru_ = node.next;
  if (idx == lru_lru_) lru_lru_ = node.prev;
  LruFreeNode(idx);
}

// 超容量时循环淘汰 LRU 尾部叶子，直到 lru_count_ <= lru_capacity_。
// 每次淘汰最久未修改的叶子：从 LRU 链表移除 → 清除 Octant 的 LRU 索引
// → 通过 BoxWiseDelete 删除该叶子覆盖的 AABB 区域。
// 在 Initialize() 和 Update() 末尾自动调用。
inline void Octree::LruEvictIfNeeded() {
  while (lru_count_ > lru_capacity_) {
    if (lru_lru_ < 0) break;
    int victim_idx = lru_lru_;
    Octant *leaf = lru_nodes_[victim_idx].octant;
    LruRemove(victim_idx);
    leaf->lru_idx_ = -1;
    float min_coord[3], max_coord[3];
    min_coord[0] = leaf->x_ - leaf->extent_;
    min_coord[1] = leaf->y_ - leaf->extent_;
    min_coord[2] = leaf->z_ - leaf->extent_;
    max_coord[0] = leaf->x_ + leaf->extent_;
    max_coord[1] = leaf->y_ + leaf->extent_;
    max_coord[2] = leaf->z_ + leaf->extent_;
    BoxWiseDelete(min_coord, max_coord, true);
  }
}

// === 节点辅助 ===

// 分配并清零 8 个子节点指针数组，将叶子转为枝干节点。
inline void Octant::InitChild() {
  child_ = new Octant *[8];
  memset(child_, 0, 8 * sizeof(Octant *));
}

// 递归累加子树中所有叶子节点的点数到 size_。
inline void Octant::GetOctantSize(Octant *octant, size_t &size_) {
  if (octant->child_ == nullptr) {
    size_ += octant->points_.size();
    return;
  }
  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    GetOctantSize(octant->child_[c], size_);
  }
}

// 返回以本节点为根的子树中的总点数。
inline size_t Octant::Size() {
  size_t pts_num = 0;
  GetOctantSize(this, pts_num);
  return pts_num;
}

// 返回当前树中的有效点数（总插入点数 - 已删除点数）。
inline size_t Octree::Size() { return last_pts_num_ - pts_num_deleted_; }

// 收集以 octant 为根的子树中所有节点。
// min_extent > 0 时，半边长 <= min_extent 的节点不再向下递归。
inline void Octree::GetNodes(Octant *octant, std::vector<Octant *> &nodes,
                             float min_extent) {
  if (octant == 0) return;
  if (min_extent > 0) {
    if (octant->extent_ <= min_extent) {
      nodes.push_back(octant);
      return;
    }
  }
  nodes.push_back(octant);
  if (octant->child_ == nullptr) {
    return;
  }
  for (int i = 0; i < 8; i++) {
    GetNodes(octant->child_[i], nodes, min_extent);
  }
}

// 收集以 octant 为根的子树中所有叶子节点。
// 深度优先遍历，叶子节点（child_ == nullptr）加入结果集。
inline void Octree::GetLeafNodes(const Octant *octant,
                                 std::vector<const Octant *> &nodes) {
  if (octant == 0) return;
  if (octant->child_ == nullptr) {
    nodes.push_back(octant);
    return;
  }
  nodes.reserve(nodes.size() + 8);
  for (int i = 0; i < 8; i++) {
    GetLeafNodes(octant->child_[i], nodes);
  }
}

// 遍历树中所有叶子节点，提取点坐标到 PointT 容器中返回。
template <typename PointT, typename ContainerT>
ContainerT Octree::GetData() {
  std::vector<Octant *> nodes;
  ContainerT pts;
  GetNodes(root_, nodes);
  for (auto octant : nodes) {
    for (auto p : octant->points_) {
      PointT pt;
      pt.x = p[0];
      pt.y = p[1];
      pt.z = p[2];
      pts.push_back(pt);
    }
  }
  return pts;
}

// === 构建 / 更新 ===

// 清空八叉树及 LRU 缓存，释放所有内存。
inline void Octree::Clear() {
  delete root_;
  root_ = 0;
  lru_free_slots_.clear();
  lru_nodes_.clear();
  lru_mru_ = lru_lru_ = -1;
  lru_count_ = 0;
}

// 从点云构建八叉树，先清空已有数据，计算包围盒后递归构建。
// 构建完成后触发 LRU 淘汰。
template <typename ContainerT>
void Octree::Initialize(ContainerT &pts_) {
  Clear();
  const size_t pts_num = pts_.size();
  std::vector<float *> points;
  points.resize(pts_num, 0);
  size_t cloud_index = 0;
  float min[3], max[3];
  for (size_t i = 0; i < pts_num; ++i) {
    const float &x = pts_[i].x;
    const float &y = pts_[i].y;
    const float &z = pts_[i].z;
    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
    float *cloud_ptr = new float[dim_];
    cloud_ptr[0] = x;
    cloud_ptr[1] = y;
    cloud_ptr[2] = z;
    if (store_index_) cloud_ptr[3] = float(cloud_index);
    points[cloud_index] = cloud_ptr;
    if (cloud_index == 0) {
      min[0] = max[0] = x;
      min[1] = max[1] = y;
      min[2] = max[2] = z;
    } else {
      min[0] = x < min[0] ? x : min[0];
      min[1] = y < min[1] ? y : min[1];
      min[2] = z < min[2] ? z : min[2];
      max[0] = x > max[0] ? x : max[0];
      max[1] = y > max[1] ? y : max[1];
      max[2] = z > max[2] ? z : max[2];
    }
    cloud_index++;
  }
  last_pts_num_ = cloud_index;
  points.resize(cloud_index);
  float ctr[3] = {min[0], min[1], min[2]};
  float max_extent = 0.5f * (max[0] - min[0]);
  max_extent = std::max(max_extent, 0.01f);
  ctr[0] += max_extent;
  for (size_t i = 1; i < 3; ++i) {
    float extent = 0.5f * (max[i] - min[i]);
    ctr[i] += extent;
    if (extent > max_extent) max_extent = extent;
  }
  root_ = CreateOctant(ctr[0], ctr[1], ctr[2], max_extent, points);
  for (size_t i = 0; i < points.size(); ++i) {
    delete[] points[i];
  }
  LruEvictIfNeeded();
}

// 递归创建八叉树节点。
// 超限（点数 > bucket_size_ 且 extent > 2*min_extent_）：按 Morton
// 码分桶递归创建子节点。 未超限：分配连续内存成为叶子，注册到 LRU。
inline Octant *Octree::CreateOctant(float x, float y, float z, float extent,
                                    std::vector<float *> &points) {
  Octant *octant = new Octant;
  const size_t size = points.size();
  octant->x_ = x;
  octant->y_ = y;
  octant->z_ = z;
  octant->extent_ = extent;
  static const float factor[] = {-0.5f, 0.5f};
  if (size > bucket_size_ && extent > 2 * min_extent_) {
    std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
    for (size_t i = 0; i < size; ++i) {
      float *p = points[i];
      size_t morton_code = 0;
      if (p[0] > x) morton_code |= 1;
      if (p[1] > y) morton_code |= 2;
      if (p[2] > z) morton_code |= 4;
      child_points[morton_code].push_back(p);
    }
    float child_extent = 0.5f * extent;
    octant->InitChild();
    for (size_t i = 0; i < 8; ++i) {
      if (child_points[i].size() == 0) continue;
      float child_x = x + factor[(i & 1) > 0] * extent;
      float child_y = y + factor[(i & 2) > 0] * extent;
      float child_z = z + factor[(i & 4) > 0] * extent;
      octant->child_[i] = CreateOctant(child_x, child_y, child_z, child_extent,
                                       child_points[i]);
    }
  } else {
    const size_t size = points.size();
    octant->points_.resize(size, 0);
    float *continue_points = new float[size * dim_];
    for (size_t i = 0; i < size; ++i) {
      std::copy(points[i], points[i] + dim_, continue_points + dim_ * i);
      octant->points_[i] = continue_points + dim_ * i;
    }
    // 新叶子注册到 LRU
    octant->lru_owner_ = this;
    int idx = LruAllocNode(octant);
    octant->lru_idx_ = idx;
    LruTouch(idx);
  }
  return octant;
}

// 增量插入新点到八叉树。
// 若树为空则调用 Initialize() 构建；否则先扩展根节点包围盒以容纳新点，
// 再通过 UpdateOctant() 递归插入，最后触发 LRU 淘汰。
template <typename ContainerT>
void Octree::Update(ContainerT &pts_) {
  if (root_ == 0) {
    Initialize(pts_);
    return;
  }
  size_t pts_num = pts_.size();
  std::vector<float *> points_tmp;
  points_tmp.resize(pts_num, 0);
  size_t cloud_index = 0;
  float min[3], max[3];
  const size_t n_old = last_pts_num_;
  for (size_t i = 0; i < pts_num; ++i) {
    const float &x = pts_[i].x;
    const float &y = pts_[i].y;
    const float &z = pts_[i].z;
    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
    float *cloud_ptr = new float[dim_];
    cloud_ptr[0] = x;
    cloud_ptr[1] = y;
    cloud_ptr[2] = z;
    if (store_index_) cloud_ptr[3] = n_old + cloud_index;
    points_tmp[cloud_index] = cloud_ptr;
    if (cloud_index == 0) {
      min[0] = max[0] = x;
      min[1] = max[1] = y;
      min[2] = max[2] = z;
    } else {
      min[0] = x < min[0] ? x : min[0];
      min[1] = y < min[1] ? y : min[1];
      min[2] = z < min[2] ? z : min[2];
      max[0] = x > max[0] ? x : max[0];
      max[1] = y > max[1] ? y : max[1];
      max[2] = z > max[2] ? z : max[2];
    }
    cloud_index++;
  }
  if (cloud_index == 0) return;
  points_tmp.resize(cloud_index);
  static const float factor[] = {-0.5f, 0.5f};
  while (std::abs(max[0] - root_->x_) > root_->extent_ ||
         std::abs(max[1] - root_->y_) > root_->extent_ ||
         std::abs(max[2] - root_->z_) > root_->extent_) {
    float parent_extent = 2 * root_->extent_;
    float parent_x = root_->x_ + factor[max[0] > root_->x_] * parent_extent;
    float parent_y = root_->y_ + factor[max[1] > root_->y_] * parent_extent;
    float parent_z = root_->z_ + factor[max[2] > root_->z_] * parent_extent;
    Octant *octant = new Octant;
    octant->x_ = parent_x;
    octant->y_ = parent_y;
    octant->z_ = parent_z;
    octant->extent_ = parent_extent;
    octant->InitChild();
    size_t morton_code = 0;
    if (root_->x_ > parent_x) morton_code |= 1;
    if (root_->y_ > parent_y) morton_code |= 2;
    if (root_->z_ > parent_z) morton_code |= 4;
    octant->child_[morton_code] = root_;
    root_ = octant;
  }
  while (std::abs(min[0] - root_->x_) > root_->extent_ ||
         std::abs(min[1] - root_->y_) > root_->extent_ ||
         std::abs(min[2] - root_->z_) > root_->extent_) {
    float parent_extent = 2 * root_->extent_;
    float parent_x = root_->x_ + factor[min[0] > root_->x_] * parent_extent;
    float parent_y = root_->y_ + factor[min[1] > root_->y_] * parent_extent;
    float parent_z = root_->z_ + factor[min[2] > root_->z_] * parent_extent;
    Octant *octant = new Octant;
    octant->x_ = parent_x;
    octant->y_ = parent_y;
    octant->z_ = parent_z;
    octant->extent_ = parent_extent;
    octant->InitChild();
    size_t morton_code = 0;
    if (root_->x_ > parent_x) morton_code |= 1;
    if (root_->y_ > parent_y) morton_code |= 2;
    if (root_->z_ > parent_z) morton_code |= 4;
    octant->child_[morton_code] = root_;
    root_ = octant;
  }

  if (points_tmp.size() == 0) return;
  last_pts_num_ += points_tmp.size();
  UpdateOctant(root_, points_tmp);
  for (size_t i = 0; i < points_tmp.size(); ++i) {
    delete[] points_tmp[i];
  }
  LruEvictIfNeeded();
}

// 增量插入新点到以 octant 为根的子树。
// 叶子超限时分裂（先从 LRU 移除，再创建子节点并注册新叶子）；
// 未超限时追加点数据并 touch MRU。
inline void Octree::UpdateOctant(Octant *octant,
                                 const std::vector<float *> &points) {
  static const float factor[] = {-0.5f, 0.5f};
  const float x = octant->x_, y = octant->y_, z = octant->z_,
              extent = octant->extent_;
  octant->is_active_ = true;
  if (octant->child_ == nullptr) {
    if (octant->points_.size() + points.size() > bucket_size_ &&
        extent > 2 * min_extent_) {
      // 分裂前从 LRU 移除（叶子变枝干）
      if (octant->lru_idx_ >= 0) {
        LruRemove(octant->lru_idx_);
        octant->lru_idx_ = -1;
      }
      octant->points_.insert(octant->points_.end(), points.begin(),
                             points.end());
      const size_t size = octant->points_.size();
      std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
      for (size_t i = 0; i < size; ++i) {
        size_t morton_code = 0;
        if (octant->points_[i][0] > x) morton_code |= 1;
        if (octant->points_[i][1] > y) morton_code |= 2;
        if (octant->points_[i][2] > z) morton_code |= 4;
        child_points[morton_code].push_back(octant->points_[i]);
      }
      float child_extent = 0.5f * extent;
      octant->InitChild();
      for (size_t i = 0; i < 8; ++i) {
        if (child_points[i].size() == 0) continue;
        float child_x = x + factor[(i & 1) > 0] * extent;
        float child_y = y + factor[(i & 2) > 0] * extent;
        float child_z = z + factor[(i & 4) > 0] * extent;
        octant->child_[i] = CreateOctant(child_x, child_y, child_z,
                                         child_extent, child_points[i]);
      }
      delete[] octant->points_[0];
      std::vector<float *>().swap(octant->points_);
    } else {
      if (octant->lru_idx_ >= 0) {
        LruTouch(octant->lru_idx_);
      }
      octant->points_.insert(octant->points_.end(), points.begin(),
                             points.end());
      const size_t size = octant->points_.size();
      float *continue_points = new float[size * dim_];
      float *old_points = octant->points_[0];
      for (size_t i = 0; i < size; ++i) {
        std::copy(octant->points_[i], octant->points_[i] + dim_,
                  continue_points + dim_ * i);
        octant->points_[i] = continue_points + dim_ * i;
      }
      delete[] old_points;
    }
  } else {
    const size_t size = points.size();
    std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
    for (size_t i = 0; i < size; ++i) {
      size_t morton_code = 0;
      if (points[i][0] > x) morton_code |= 1;
      if (points[i][1] > y) morton_code |= 2;
      if (points[i][2] > z) morton_code |= 4;
      child_points[morton_code].push_back(points[i]);
    }
    float child_extent = 0.5f * extent;
    for (size_t i = 0; i < 8; ++i) {
      if (child_points[i].size() > 0) {
        if (octant->child_[i] == 0) {
          float child_x = x + factor[(i & 1) > 0] * extent;
          float child_y = y + factor[(i & 2) > 0] * extent;
          float child_z = z + factor[(i & 4) > 0] * extent;
          octant->child_[i] = CreateOctant(child_x, child_y, child_z,
                                           child_extent, child_points[i]);
        } else
          UpdateOctant(octant->child_[i], child_points[i]);
      }
    }
  }
}

// === 删除 ===

// 递归删除 AABB 区域内的节点。
// 三种情况处理：octant 完全在 AABB
// 内则整棵子树删除；叶子部分覆盖则逐点过滤保留外部点；
// 枝干部分覆盖则递归子节点，子节点全部删除后级联删除当前节点。
// deleted 输出参数指示当前 octant 是否被完全删除。
inline void Octree::BoxWiseDelete(Octant *octant,
                                  const BoxDeleteType &box_range, bool &deleted,
                                  bool clear_data) {
  float cur_min[3];
  float cur_max[3];
  cur_min[0] = octant->x_ - octant->extent_;
  cur_min[1] = octant->y_ - octant->extent_;
  cur_min[2] = octant->z_ - octant->extent_;
  cur_max[0] = octant->x_ + octant->extent_;
  cur_max[1] = octant->y_ + octant->extent_;
  cur_max[2] = octant->z_ + octant->extent_;
  if (cur_min[0] > box_range.max[0] || box_range.min[0] > cur_max[0]) return;
  if (cur_min[1] > box_range.max[1] || box_range.min[1] > cur_max[1]) return;
  if (cur_min[2] > box_range.max[2] || box_range.min[2] > cur_max[2]) return;
  if (cur_min[0] >= box_range.min[0] && cur_min[1] >= box_range.min[1] &&
      cur_min[2] >= box_range.min[2] && cur_max[0] <= box_range.max[0] &&
      cur_max[1] <= box_range.max[1] && cur_max[2] <= box_range.max[2]) {
    if (!clear_data) {
      octant->is_active_ = false;
      return;
    } else {
      pts_num_deleted_ += octant->Size();
      delete octant;
      deleted = true;
      return;
    }
  }
  if (octant->child_ == nullptr) {
    if (!clear_data) {
      octant->is_active_ = false;
      return;
    } else {
      const size_t size = octant->points_.size();
      std::vector<float *> remainder_points;
      remainder_points.resize(size, 0);
      size_t valid_num = 0;
      for (int i = 0; i < size; ++i) {
        const float *p = octant->points_[i];
        if (p[0] > box_range.max[0] || box_range.min[0] > p[0]) {
          remainder_points[valid_num] = octant->points_[i];
          valid_num++;
          continue;
        }
        if (p[1] > box_range.max[1] || box_range.min[1] > p[1]) {
          remainder_points[valid_num] = octant->points_[i];
          valid_num++;
          continue;
        }
        if (p[2] > box_range.max[2] || box_range.min[2] > p[2]) {
          remainder_points[valid_num] = octant->points_[i];
          valid_num++;
          continue;
        }
      }
      pts_num_deleted_ += size - valid_num;
      if (valid_num == 0) {
        delete octant;
        deleted = true;
        return;
      }
      float *continue_points = new float[valid_num * dim_];
      float *old_points = octant->points_[0];
      for (size_t i = 0; i < valid_num; ++i) {
        std::copy(remainder_points[i], remainder_points[i] + dim_,
                  continue_points + dim_ * i);
        octant->points_[i] = continue_points + dim_ * i;
      }
      octant->points_.resize(valid_num);
      delete[] old_points;
      return;
    }
  }

  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    bool deleted = false;
    BoxWiseDelete(octant->child_[c], box_range, deleted, clear_data);
    if (deleted) octant->child_[c] = 0;
  }
  int valid_child = 0;
  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    valid_child++;
  }
  if (valid_child == 0) {
    delete octant;
    deleted = true;
    return;
  }
}

// 从根节点删除 AABB 区域内的节点。
// clear_data=true 时物理删除点数据，false 时仅标记 is_active_。
inline void Octree::BoxWiseDelete(const BoxDeleteType &box_range,
                                  bool clear_data) {
  if (root_ == 0) return;
  bool deleted = false;
  BoxWiseDelete(root_, box_range, deleted, clear_data);
  if (deleted) root_ = 0;
}

// 从根节点删除由对角点 [min, max] 定义的 AABB 区域内的节点。
inline void Octree::BoxWiseDelete(const float *min, const float *max,
                                  bool clear_data) {
  if (root_ == 0) return;
  BoxDeleteType box_range;
  box_range.min[0] = min[0];
  box_range.min[1] = min[1];
  box_range.min[2] = min[2];
  box_range.max[0] = max[0];
  box_range.max[1] = max[1];
  box_range.max[2] = max[2];
  bool deleted = false;
  BoxWiseDelete(root_, box_range, deleted, clear_data);
  if (deleted) root_ = 0;
}

// === 几何测试 ===

// 判断搜索球 S(q,r) 与 octant 的 AABB 是否有交集。
// 利用对称性将测试简化为正象限的 Minkowski 和判断：分别对 x/y/z 方向检查球
// 与 octant 的投影区间是否相交，三个方向都相交才有交集。
inline bool Octree::Overlaps(const float *query, float sqr_radius,
                             const Octant *o) {
  float x = std::abs(query[0] - o->x_) - o->extent_;
  float y = std::abs(query[1] - o->y_) - o->extent_;
  float z = std::abs(query[2] - o->z_) - o->extent_;
  if ((x > 0 && x * x > sqr_radius) || (y > 0 && y * y > sqr_radius) ||
      (z > 0 && z * z > sqr_radius))
    return false;
  int32_t num_less_extent = (x < 0) + (y < 0) + (z < 0);
  if (num_less_extent > 1) return true;
  x = std::max(x, 0.0f);
  y = std::max(y, 0.0f);
  z = std::max(z, 0.0f);
  return (x * x + y * y + z * z < sqr_radius);
}

// 判断搜索球 S(q,r) 是否完全包住 octant 的 AABB。
// 取 octant 最远角点，若该点仍在球内则整个 octant 被球包含。
inline bool Octree::Contains(const float *query, float sqr_radius,
                             const Octant *o) {
  float x = std::abs(query[0] - o->x_) + o->extent_;
  float y = std::abs(query[1] - o->y_) + o->extent_;
  float z = std::abs(query[2] - o->z_) + o->extent_;
  return (x * x + y * y + z * z < sqr_radius);
}

// 判断搜索球 S(q,r) 是否完全在 octant 的 AABB 内部。
// 用于 KNN 早停：若球完全在节点内，子树中所有更优解均已包含在结果集中。
inline bool Octree::Inside(const float *query, float sqr_radius,
                           const Octant *octant) {
  float x = octant->extent_ - std::abs(query[0] - octant->x_);
  float y = octant->extent_ - std::abs(query[1] - octant->y_);
  float z = octant->extent_ - std::abs(query[2] - octant->z_);
  if (x < 0 || x * x < sqr_radius) return false;
  if (y < 0 || y * y < sqr_radius) return false;
  if (z < 0 || z * z < sqr_radius) return false;
  return true;
}

// 计算查询点到 octant AABB 的平方距离，点在内部或边界上时返回 0。
// === 搜索 — 内部实现 ===

// 半径搜索递归实现，仅返回点坐标指针（不含距离）。
// 逻辑同带距离版本，但不计算和返回平方距离。
inline void Octree::RadiusNeighbors(const Octant *octant, const float *query,
                                    float radius, float sqr_radius,
                                    std::vector<float *> &result_indices) {
  if (!octant->is_active_) return;
  if (3 * octant->extent_ * octant->extent_ < sqr_radius &&
      Contains(query, sqr_radius, octant)) {
    std::vector<const Octant *> candidate_octants;
    candidate_octants.reserve(8);
    GetLeafNodes(octant, candidate_octants);
    for (size_t k = 0; k < candidate_octants.size(); k++) {
      const size_t size = candidate_octants[k]->points_.size();
      const size_t result_size = result_indices.size();
      result_indices.resize(result_size + size);
      for (size_t i = 0; i < size; ++i) {
        result_indices[result_size + i] = candidate_octants[k]->points_[i];
      }
    }
    return;
  }
  if (octant->child_ == nullptr) {
    const size_t size = octant->points_.size();
    for (size_t i = 0; i < size; ++i) {
      const float *p = octant->points_[i];
      float dist = 0, diff = 0;
      for (size_t j = 0; j < 3; ++j) {
        diff = p[j] - query[j];
        dist += diff * diff;
      }
      if (dist < sqr_radius) result_indices.push_back(octant->points_[i]);
    }
    return;
  }
  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    if (!Overlaps(query, sqr_radius, octant->child_[c])) continue;
    RadiusNeighbors(octant->child_[c], query, radius, sqr_radius,
                    result_indices);
  }
}

// 半径搜索递归实现，返回点坐标指针和对应距离。
// 若 octant 被搜索球完全包住（Contains），直接收集该子树所有叶子点。
// 否则递归子节点，通过 Overlaps() 剪枝跳过不相交的子节点。
inline void Octree::RadiusNeighbors(const Octant *octant, const float *query,
                                    float radius, float sqr_radius,
                                    std::vector<float *> &result_indices,
                                    std::vector<float> &distances) {
  if (!octant->is_active_) return;
  if (3 * octant->extent_ * octant->extent_ < sqr_radius &&
      Contains(query, sqr_radius, octant)) {
    std::vector<const Octant *> candidate_octants;
    GetLeafNodes(octant, candidate_octants);
    for (size_t k = 0; k < candidate_octants.size(); k++) {
      const size_t size = candidate_octants[k]->points_.size();
      const size_t result_size = result_indices.size();
      result_indices.resize(result_size + size);
      for (size_t i = 0; i < size; ++i) {
        const float *p = candidate_octants[k]->points_[i];
        float dist = 0, diff = 0;
        for (size_t j = 0; j < 3; ++j) {
          diff = p[j] - query[j];
          dist += diff * diff;
        }
        distances.push_back(dist);
        result_indices[result_size + i] = candidate_octants[k]->points_[i];
      }
    }
    return;
  }
  if (octant->child_ == nullptr) {
    const size_t size = octant->points_.size();
    for (size_t i = 0; i < size; ++i) {
      const float *p = octant->points_[i];
      float dist = 0, diff = 0;
      for (size_t j = 0; j < 3; ++j) {
        diff = p[j] - query[j];
        dist += diff * diff;
      }
      if (dist < sqr_radius) {
        result_indices.push_back(octant->points_[i]);
        distances.push_back(dist);
      }
    }
    return;
  }
  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    if (!Overlaps(query, sqr_radius, octant->child_[c])) continue;
    RadiusNeighbors(octant->child_[c], query, radius, sqr_radius,
                    result_indices, distances);
  }
}

// 半径搜索递归实现（带索引）。
// 沿 Morton 码优先路径下降，通过 Overlaps() 剪枝和 Inside() 早停。
// 返回 true 表示搜索球完全在节点内。
inline bool Octree::RadiusNeighborsFast(const Octant *octant, const float *query,
                                     float sqr_radius,
                                     std::vector<size_t> &result_indices,
                                     std::vector<float> &distances) {
  if (!octant->is_active_) return false;
  if (octant->child_ == nullptr) {
    const size_t size = octant->points_.size();
    for (int i = 0; i < size; ++i) {
      const float *p = octant->points_[i];
      float dist = 0, diff = 0;
      for (int j = 0; j < 3; ++j) {
        diff = p[j] - query[j];
        dist += diff * diff;
      }
      if (dist < sqr_radius) {
        result_indices.push_back(size_t(p[3]));
        distances.push_back(dist);
      }
    }
    return Inside(query, sqr_radius, octant);
  }
  size_t morton_code = 0;
  if (query[0] > octant->x_) morton_code |= 1;
  if (query[1] > octant->y_) morton_code |= 2;
  if (query[2] > octant->z_) morton_code |= 4;
  if (octant->child_[morton_code] != 0) {
    if (RadiusNeighborsFast(octant->child_[morton_code], query, sqr_radius,
                         result_indices, distances))
      return true;
  }
  for (int i = 0; i < 7; ++i) {
    int c = ordered_indices_[morton_code][i];
    if (octant->child_[c] == 0) continue;
    if (!Overlaps(query, sqr_radius, octant->child_[c])) continue;
    if (RadiusNeighborsFast(octant->child_[c], query, sqr_radius, result_indices,
                         distances))
      return true;
  }
  return Inside(query, sqr_radius, octant);
}

// KNN 搜索核心递归。
// 沿 Morton 码优先路径从根下降到查询点所在叶子，堆满后通过 Overlaps()
// 剪枝跳过不相交的子节点，通过 Inside() 判断搜索球完全在节点内时早停。
// 返回 true 表示搜索球完全在当前 octant 内，调用方可跳过其余回溯。
inline bool Octree::KnnNeighbors(const Octant *octant, const float *query,
                                 KNNSimpleResultSet &heap) {
  if (!octant->is_active_) return false;
  if (octant->child_ == nullptr) {
    const size_t size = octant->points_.size();
    for (int i = 0; i < size; ++i) {
      const float *p = octant->points_[i];
      float dist = 0, diff = 0;
      for (int j = 0; j < 3; ++j) {
        diff = p[j] - query[j];
        dist += diff * diff;
      }
      if (dist < heap.WorstDist()) heap.AddPoint(dist, octant->points_[i]);
    }
    return heap.Full() && Inside(query, heap.WorstDist(), octant);
  }
  size_t morton_code = 0;
  if (query[0] > octant->x_) morton_code |= 1;
  if (query[1] > octant->y_) morton_code |= 2;
  if (query[2] > octant->z_) morton_code |= 4;
  if (octant->child_[morton_code] != 0) {
    if (KnnNeighbors(octant->child_[morton_code], query, heap)) return true;
  }
  for (int i = 0; i < 7; ++i) {
    int c = ordered_indices_[morton_code][i];
    if (octant->child_[c] == 0) continue;
    if (heap.Full() && !Overlaps(query, heap.WorstDist(), octant->child_[c]))
      continue;
    if (KnnNeighbors(octant->child_[c], query, heap)) return true;
  }
  return heap.Full() && Inside(query, heap.WorstDist(), octant);
}

// 递归 AABB 范围搜索，收集指定包围盒内的所有点数据指针。
inline void Octree::SearchByBox(const Octant *octant, const float *min,
                                const float *max,
                                std::vector<float *> &result_indices) {
  if (!octant->is_active_) return;
  float cur_min[3] = {octant->x_ - octant->extent_,
                      octant->y_ - octant->extent_,
                      octant->z_ - octant->extent_};
  float cur_max[3] = {octant->x_ + octant->extent_,
                      octant->y_ + octant->extent_,
                      octant->z_ + octant->extent_};
  if (cur_min[0] > max[0] || min[0] > cur_max[0]) return;
  if (cur_min[1] > max[1] || min[1] > cur_max[1]) return;
  if (cur_min[2] > max[2] || min[2] > cur_max[2]) return;
  if (cur_min[0] >= min[0] && cur_min[1] >= min[1] && cur_min[2] >= min[2] &&
      cur_max[0] <= max[0] && cur_max[1] <= max[1] && cur_max[2] <= max[2]) {
    std::vector<const Octant *> leaf_nodes;
    GetLeafNodes(octant, leaf_nodes);
    for (size_t k = 0; k < leaf_nodes.size(); k++) {
      const size_t size = leaf_nodes[k]->points_.size();
      const size_t result_size = result_indices.size();
      result_indices.resize(result_size + size);
      for (size_t i = 0; i < size; ++i) {
        result_indices[result_size + i] = leaf_nodes[k]->points_[i];
      }
    }
    return;
  }
  if (octant->child_ == nullptr) {
    const size_t size = octant->points_.size();
    for (size_t i = 0; i < size; ++i) {
      const float *p = octant->points_[i];
      if (p[0] >= min[0] && p[0] <= max[0] && p[1] >= min[1] && p[1] <= max[1] &&
          p[2] >= min[2] && p[2] <= max[2]) {
        result_indices.push_back(octant->points_[i]);
      }
    }
    return;
  }
  for (size_t c = 0; c < 8; ++c) {
    if (octant->child_[c] == 0) continue;
    SearchByBox(octant->child_[c], min, max, result_indices);
  }
}

// === 搜索 — 公开接口 ===

// 半径搜索：返回查询点半径范围内所有点的原始索引（需 store_index_=true）。
template <typename PointT>
void Octree::RadiusNeighbors(const PointT &query_in, float radius,
                             std::vector<size_t> &result_indices) {
  result_indices.clear();
  if (!store_index_) return;
  if (root_ == 0) return;
  float sqr_radius = radius * radius;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  std::vector<float *> points_ptr;
  RadiusNeighbors(root_, query, radius, sqr_radius, points_ptr);
  result_indices.resize(points_ptr.size());
  for (size_t i = 0; i < points_ptr.size(); i++) {
    result_indices[i] = size_t(points_ptr[i][3]);
  }
}

// 半径搜索：返回查询点半径范围内所有点的原始索引和平方距离（需
// store_index_=true）。
template <typename PointT>
void Octree::RadiusNeighbors(const PointT &query_in, float radius,
                             std::vector<size_t> &result_indices,
                             std::vector<float> &distances) {
  result_indices.clear();
  distances.clear();
  if (!store_index_) return;
  if (root_ == 0) return;
  float sqr_radius = radius * radius;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  std::vector<float *> points_ptr;
  RadiusNeighbors(root_, query, radius, sqr_radius, points_ptr, distances);
  result_indices.resize(points_ptr.size());
  for (size_t i = 0; i < points_ptr.size(); i++) {
    result_indices[i] = size_t(points_ptr[i][3]);
  }
}

// 半径搜索：返回查询点半径范围内所有点的坐标和平方距离。
template <typename PointT>
void Octree::RadiusNeighbors(const PointT &query_in, float radius,
                             std::vector<PointT> &result_indices,
                             std::vector<float> &distances) {
  result_indices.clear();
  distances.clear();
  if (root_ == 0) return;
  float sqr_radius = radius * radius;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  std::vector<float *> points_ptr;
  RadiusNeighbors(root_, query, radius, sqr_radius, points_ptr, distances);
  result_indices.resize(points_ptr.size());
  for (size_t i = 0; i < result_indices.size(); i++) {
    PointT pt;
    pt.x = points_ptr[i][0];
    pt.y = points_ptr[i][1];
    pt.z = points_ptr[i][2];
    result_indices[i] = pt;
  }
}

// 半径搜索：返回查询点半径范围内所有点的坐标。
template <typename PointT>
void Octree::RadiusNeighbors(const PointT &query_in, float radius,
                             std::vector<PointT> &result_indices) {
  result_indices.clear();
  if (root_ == 0) return;
  float sqr_radius = radius * radius;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  std::vector<float *> points_ptr;
  RadiusNeighbors(root_, query, radius, sqr_radius, points_ptr);
  result_indices.resize(points_ptr.size());
  for (size_t i = 0; i < result_indices.size(); i++) {
    PointT pt;
    pt.x = points_ptr[i][0];
    pt.y = points_ptr[i][1];
    pt.z = points_ptr[i][2];
    result_indices[i] = pt;
  }
}

// KNN 搜索：返回最近的 k 个点的坐标和平方距离。
// 构造堆后从根节点递归搜索，结果按距离升序排列。
template <typename PointT>
int32_t Octree::KnnNeighbors(const PointT &query_in, int k,
                             std::vector<PointT> &result_indices,
                             std::vector<float> &distances) {
  if (root_ == 0) return 0;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  KNNSimpleResultSet heap(k);
  KnnNeighbors(root_, query, heap);
  const std::vector<DistanceIndex> &data = heap.GetData();
  result_indices.resize(heap.Size());
  distances.resize(heap.Size());
  for (size_t i = 0; i < heap.Size(); i++) {
    PointT pt;
    pt.x = data[i].index[0];
    pt.y = data[i].index[1];
    pt.z = data[i].index[2];
    result_indices[i] = pt;
    distances[i] = data[i].dist;
  }
  return data.size();
}

// KNN 搜索：返回最近的 k 个点的原始索引和平方距离（需 store_index_=true）。
template <typename PointT>
int32_t Octree::KnnNeighbors(const PointT &query_in, int k,
                             std::vector<size_t> &result_indices,
                             std::vector<float> &distances) {
  if (!store_index_) return 0;
  if (root_ == 0) return 0;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  KNNSimpleResultSet heap(k);
  KnnNeighbors(root_, query, heap);
  const std::vector<DistanceIndex> &data = heap.GetData();
  result_indices.resize(heap.Size());
  distances.resize(heap.Size());
  for (size_t i = 0; i < heap.Size(); i++) {
    result_indices[i] = size_t(data[i].index[3]);
    distances[i] = data[i].dist;
  }
  return data.size();
}

// KNN 搜索：返回最近的 k 个点的原始索引(int)和平方距离（需
// store_index_=true）。
template <typename PointT>
int32_t Octree::KnnNeighbors(const PointT &query_in, int k,
                             std::vector<int> &result_indices,
                             std::vector<float> &distances) {
  if (!store_index_) return 0;
  if (root_ == 0) return 0;
  float query[3] = {query_in.x, query_in.y, query_in.z};
  KNNSimpleResultSet heap(k);
  KnnNeighbors(root_, query, heap);
  const std::vector<DistanceIndex> &data = heap.GetData();
  result_indices.resize(heap.Size());
  distances.resize(heap.Size());
  for (size_t i = 0; i < heap.Size(); i++) {
    result_indices[i] = int(data[i].index[3]);
    distances[i] = data[i].dist;
  }
  return data.size();
}

// 按 AABB 对角范围搜索，返回范围内所有点的坐标。
template <typename ContainerT>
void Octree::SearchByBox(const float *min, const float *max,
                         ContainerT &result_points) {
  result_points.clear();
  if (root_ == 0) return;
  std::vector<float *> points_ptr;
  SearchByBox(root_, min, max, points_ptr);
  result_points.resize(points_ptr.size());
  using PointType = typename ContainerT::value_type;
  for (size_t i = 0; i < points_ptr.size(); i++) {
    PointType pt;
    pt.x = points_ptr[i][0];
    pt.y = points_ptr[i][1];
    pt.z = points_ptr[i][2];
    result_points[i] = pt;
  }
}

// === 记录 / 调试 ===

// 从点云构建八叉树并记录节点信息（调试/分析用）。
// 先清空已有数据，计算包围盒，再递归构建并在创建节点时通过 info_record_ 记录。
template <typename ContainerT>
void Octree::InitializeRecord(ContainerT &points) {
  Clear();
  const size_t pts_num = points.size();
  std::vector<float *> points_vec;
  points_vec.resize(pts_num, 0);
  size_t cloud_index = 0;
  float min[3], max[3];
  for (size_t i = 0; i < pts_num; ++i) {
    const float &x = points[i].x;
    const float &y = points[i].y;
    const float &z = points[i].z;
    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
    float *cloud_ptr = new float[dim_];
    cloud_ptr[0] = x;
    cloud_ptr[1] = y;
    cloud_ptr[2] = z;
    if (store_index_) cloud_ptr[3] = float(cloud_index);
    points_vec[cloud_index] = cloud_ptr;
    if (cloud_index == 0) {
      min[0] = max[0] = x;
      min[1] = max[1] = y;
      min[2] = max[2] = z;
    } else {
      min[0] = x < min[0] ? x : min[0];
      min[1] = y < min[1] ? y : min[1];
      min[2] = z < min[2] ? z : min[2];
      max[0] = x > max[0] ? x : max[0];
      max[1] = y > max[1] ? y : max[1];
      max[2] = z > max[2] ? z : max[2];
    }
    cloud_index++;
  }
  last_pts_num_ = cloud_index;
  points.resize(cloud_index);
  float ctr[3] = {min[0], min[1], min[2]};
  float max_extent = 0.5f * (max[0] - min[0]);
  max_extent = std::max(max_extent, 0.01f);
  ctr[0] += max_extent;
  for (size_t i = 1; i < 3; ++i) {
    float extent = 0.5f * (max[i] - min[i]);
    ctr[i] += extent;
    if (extent > max_extent) max_extent = extent;
  }
  root_ = CreateOctantRecord(ctr[0], ctr[1], ctr[2], max_extent, points_vec);
  for (size_t i = 0; i < points_vec.size(); ++i) {
    delete[] points_vec[i];
  }
}

// 递归创建八叉树节点并注册到 info_record_（调试/分析用）。
// 超限时按 Morton 码分桶递归创建子节点；未超限则分配连续内存成为叶子。
inline Octant *Octree::CreateOctantRecord(float x, float y, float z,
                                          float extent,
                                          std::vector<float *> &points) {
  Octant *octant = new Octant;
  const size_t size = points.size();
  octant->x_ = x;
  octant->y_ = y;
  octant->z_ = z;
  octant->extent_ = extent;
  info_record_.AddOctant(octant);
  static const float factor[] = {-0.5f, 0.5f};
  if (size > bucket_size_ && extent > 2 * min_extent_) {
    std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
    for (size_t i = 0; i < size; ++i) {
      float *p = points[i];
      size_t morton_code = 0;
      if (p[0] > x) morton_code |= 1;
      if (p[1] > y) morton_code |= 2;
      if (p[2] > z) morton_code |= 4;
      child_points[morton_code].push_back(p);
    }
    float child_extent = 0.5f * extent;
    octant->InitChild();
    for (size_t i = 0; i < 8; ++i) {
      if (child_points[i].size() == 0) continue;
      float child_x = x + factor[(i & 1) > 0] * extent;
      float child_y = y + factor[(i & 2) > 0] * extent;
      float child_z = z + factor[(i & 4) > 0] * extent;
      octant->child_[i] = CreateOctantRecord(child_x, child_y, child_z,
                                             child_extent, child_points[i]);
    }
  } else {
    const size_t size = points.size();
    octant->points_.resize(size, 0);
    float *continue_points = new float[size * dim_];
    for (size_t i = 0; i < size; ++i) {
      std::copy(points[i], points[i] + dim_, continue_points + dim_ * i);
      octant->points_[i] = continue_points + dim_ * i;
    }
  }
  return octant;
}

// 增量插入新点到八叉树（带 info_record 记录）。
// 若树为空则调用 InitializeRecord；否则扩展根节点包围盒后插入，在创建父节点
// 和新子节点时通过 info_record_ 记录。
template <typename ContainerT>
void Octree::UpdateRecord(ContainerT &points) {
  if (root_ == 0) {
    InitializeRecord(points);
    return;
  }
  size_t pts_num = points.size();
  std::vector<float *> points_tmp;
  points_tmp.resize(pts_num, 0);
  size_t cloud_index = 0;
  float min[3], max[3];
  const size_t n_old = last_pts_num_;
  for (size_t i = 0; i < pts_num; ++i) {
    const float &x = points[i].x;
    const float &y = points[i].y;
    const float &z = points[i].z;
    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
    float *cloud_ptr = new float[dim_];
    cloud_ptr[0] = x;
    cloud_ptr[1] = y;
    cloud_ptr[2] = z;
    if (store_index_) cloud_ptr[3] = n_old + cloud_index;
    points_tmp[cloud_index] = cloud_ptr;
    if (cloud_index == 0) {
      min[0] = max[0] = x;
      min[1] = max[1] = y;
      min[2] = max[2] = z;
    } else {
      min[0] = x < min[0] ? x : min[0];
      min[1] = y < min[1] ? y : min[1];
      min[2] = z < min[2] ? z : min[2];
      max[0] = x > max[0] ? x : max[0];
      max[1] = y > max[1] ? y : max[1];
      max[2] = z > max[2] ? z : max[2];
    }
    cloud_index++;
  }
  if (cloud_index == 0) return;
  points_tmp.resize(cloud_index);
  static const float factor[] = {-0.5f, 0.5f};
  while (std::abs(max[0] - root_->x_) > root_->extent_ ||
         std::abs(max[1] - root_->y_) > root_->extent_ ||
         std::abs(max[2] - root_->z_) > root_->extent_) {
    float parent_extent = 2 * root_->extent_;
    float parent_x = root_->x_ + factor[max[0] > root_->x_] * parent_extent;
    float parent_y = root_->y_ + factor[max[1] > root_->y_] * parent_extent;
    float parent_z = root_->z_ + factor[max[2] > root_->z_] * parent_extent;
    Octant *octant = new Octant;
    octant->x_ = parent_x;
    octant->y_ = parent_y;
    octant->z_ = parent_z;
    octant->extent_ = parent_extent;
    octant->InitChild();
    size_t morton_code = 0;
    if (root_->x_ > parent_x) morton_code |= 1;
    if (root_->y_ > parent_y) morton_code |= 2;
    if (root_->z_ > parent_z) morton_code |= 4;
    octant->child_[morton_code] = root_;
    root_ = octant;
    info_record_.AddOctant(octant);
  }
  while (std::abs(min[0] - root_->x_) > root_->extent_ ||
         std::abs(min[1] - root_->y_) > root_->extent_ ||
         std::abs(min[2] - root_->z_) > root_->extent_) {
    float parent_extent = 2 * root_->extent_;
    float parent_x = root_->x_ + factor[min[0] > root_->x_] * parent_extent;
    float parent_y = root_->y_ + factor[min[1] > root_->y_] * parent_extent;
    float parent_z = root_->z_ + factor[min[2] > root_->z_] * parent_extent;
    Octant *octant = new Octant;
    octant->x_ = parent_x;
    octant->y_ = parent_y;
    octant->z_ = parent_z;
    octant->extent_ = parent_extent;
    octant->InitChild();
    size_t morton_code = 0;
    if (root_->x_ > parent_x) morton_code |= 1;
    if (root_->y_ > parent_y) morton_code |= 2;
    if (root_->z_ > parent_z) morton_code |= 4;
    octant->child_[morton_code] = root_;
    root_ = octant;
    info_record_.AddOctant(octant);
  }

  if (points_tmp.size() == 0) return;
  last_pts_num_ += points_tmp.size();
  UpdateOctantRecord(root_, points_tmp);
  for (size_t i = 0; i < points_tmp.size(); ++i) {
    delete[] points_tmp[i];
  }
}

// 增量插入新点到子树，并在分裂/创建子节点时通过 info_record_ 记录节点信息。
// 逻辑与 UpdateOctant 一致，额外在创建新节点时调用 info_record_.AddOctant()。
inline void Octree::UpdateOctantRecord(Octant *octant,
                                       const std::vector<float *> &points) {
  static const float factor[] = {-0.5f, 0.5f};
  const float x = octant->x_, y = octant->y_, z = octant->z_,
              extent = octant->extent_;
  octant->is_active_ = true;
  if (octant->child_ == nullptr) {
    if (octant->points_.size() + points.size() > bucket_size_ &&
        extent > 2 * min_extent_) {
      octant->points_.insert(octant->points_.end(), points.begin(),
                             points.end());
      const size_t size = octant->points_.size();
      std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
      for (size_t i = 0; i < size; ++i) {
        size_t morton_code = 0;
        if (octant->points_[i][0] > x) morton_code |= 1;
        if (octant->points_[i][1] > y) morton_code |= 2;
        if (octant->points_[i][2] > z) morton_code |= 4;
        child_points[morton_code].push_back(octant->points_[i]);
      }
      float child_extent = 0.5f * extent;
      octant->InitChild();
      for (size_t i = 0; i < 8; ++i) {
        if (child_points[i].size() == 0) continue;
        float child_x = x + factor[(i & 1) > 0] * extent;
        float child_y = y + factor[(i & 2) > 0] * extent;
        float child_z = z + factor[(i & 4) > 0] * extent;
        octant->child_[i] = CreateOctantRecord(child_x, child_y, child_z,
                                               child_extent, child_points[i]);
      }
      delete[] octant->points_[0];
      std::vector<float *>().swap(octant->points_);
    } else {
      octant->points_.insert(octant->points_.end(), points.begin(),
                             points.end());
      const size_t size = octant->points_.size();
      float *continue_points = new float[size * dim_];
      float *old_points = octant->points_[0];
      for (size_t i = 0; i < size; ++i) {
        std::copy(octant->points_[i], octant->points_[i] + dim_,
                  continue_points + dim_ * i);
        octant->points_[i] = continue_points + dim_ * i;
      }
      delete[] old_points;
    }
  } else {
    const size_t size = points.size();
    std::vector<std::vector<float *>> child_points(8, std::vector<float *>());
    for (size_t i = 0; i < size; ++i) {
      size_t morton_code = 0;
      if (points[i][0] > x) morton_code |= 1;
      if (points[i][1] > y) morton_code |= 2;
      if (points[i][2] > z) morton_code |= 4;
      child_points[morton_code].push_back(points[i]);
    }
    float child_extent = 0.5f * extent;
    for (size_t i = 0; i < 8; ++i) {
      if (child_points[i].size() > 0) {
        if (octant->child_[i] == 0) {
          float child_x = x + factor[(i & 1) > 0] * extent;
          float child_y = y + factor[(i & 2) > 0] * extent;
          float child_z = z + factor[(i & 4) > 0] * extent;
          octant->child_[i] = CreateOctantRecord(child_x, child_y, child_z,
                                                 child_extent, child_points[i]);
        } else
          UpdateOctantRecord(octant->child_[i], child_points[i]);
      }
    }
  }
}

// 将 info_record_ 中收集的节点信息写出到文本文件。
// clear=true 时写出后清空记录。
inline void Octree::WriteInfoRecord(std::string name, bool clear) {
  info_record_.WriteToTxt(name);
  if (clear) {
    info_record_.Clear();
  }
}

// 将当前树中所有节点信息写出到文本文件（调试/分析用）。
inline void Octree::WriteRemainNodeRecord(std::string name) {
  if (root_ == 0) return;
  std::vector<Octant *> nodes;
  GetNodes(root_, nodes);
  {
    std::string filename =
        "/media/zhujun/0DFD06D20DFD06D2/SLAM/octree_test/ikd-Tree-main/data/";
    filename += name + ".txt";
    std::ofstream log_file(filename, std::ios::out);
    log_file << "# x y z extent" << std::endl;
    for (int i = 0; i < nodes.size(); i++) {
      log_file << nodes[i]->x_ << " " << nodes[i]->y_ << " " << nodes[i]->z_
               << " " << nodes[i]->extent_ << "\n";
    }
    log_file.close();
  }
}

// 将当前树中所有节点信息写入固定路径的文本文件（调试用）。
inline void Octree::WriteNodeInfoToTxt(float min_extent) {
  if (root_ == 0) return;
  std::vector<Octant *> nodes;
  GetNodes(root_, nodes);
  {
    std::ofstream log_file(
        "/media/zhujun/0DFD06D20DFD06D2/SLAM/octree_test/ikd-Tree-main/data/"
        "node_info.txt",
        std::ios::out);
    log_file << "# x y z extent" << std::endl;
    for (int i = 0; i < nodes.size(); i++) {
      log_file << nodes[i]->x_ << " " << nodes[i]->y_ << " " << nodes[i]->z_
               << " " << nodes[i]->extent_ << "\n";
    }
    log_file.close();
  }
}

}  // namespace thuni

#endif /* THUNI_OCTREE_HPP_ */