#include "hmt/bc_label.hxx"
#include "hmt/tree_build.hxx"
#include "np_helpers.hxx"
#include "pyglia.hxx"
#include "type/big_num.hxx"
#include "type/region_map.hxx"
#include "type/tuple.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/mp.hxx"
#include "util/text_cmd.hxx"
#include "util/text_io.hxx"

using namespace glia;
using namespace glia::hmt;

using LabelImageType = LabelImage<DIMENSION>;
using RealImageType = RealImage<DIMENSION>;

struct NodeData {
  Label label;
  std::vector<int> bestSplits;
  int bcLabel = BC_LABEL_MERGE;
};

// This corresponds to section III-C of the paper

// For a given merge order and saliency metric, compute for each clique
// a merge(+1)/split(-1) label

// Parameters

// Input initial segmentation image file name")
// Input merging order file name")
// Input ground truth segmentation image file name")
// Input mask image file name (optional)")
// Whether to use pair F1 other than traditional Rand index "
//[default: true]")
// Global optimal assignment type (0: none, 1: RCC by merge, "
// 2: RCC by split) [default: 0]")
// Whether to determine based on optimal splits [default: false]")
// Whether to tweak conditions for thick boundaries [default: false]")
// Maximum precision drop allowed for merge [default: 1.0]")

std::vector<int> bc_label_ri_operation(
    std::vector<TTriple<Label>> const &order, LabelImageType::Pointer labels,
    LabelImageType::Pointer groundtruth, bool const &usePairF1,
    int const &globalOpt, bool const &optSplit, bool const &tweak,
    double const &maxPrecDrop) {

  LabelImageType::Pointer mask = LabelImageType::Pointer(nullptr);

  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(labels, mask, order, false);
  int n = order.size();
  std::vector<int> bcLabels;
  if (globalOpt == 0) { // Local optimal
    bcLabels.resize(n);
    if (usePairF1) { // Use pair F1
      std::vector<double> mergeF1s, splitF1s;
      mergeF1s.resize(n);
      splitF1s.resize(n);
      parfor(
          0, n, true,
          [&bcLabels, &mergeF1s, &splitF1s, &rmap, &order, &groundtruth, &tweak,
           &maxPrecDrop](int i) {
            bcLabels[i] = genBoundaryClassificationLabelF1<BigInt>(
                mergeF1s[i], splitF1s[i], rmap, order[i].x0, order[i].x1,
                order[i].x2, groundtruth, tweak, maxPrecDrop);
          },
          0);
      if (optSplit) {
        std::unordered_map<Label, std::vector<RegionMap::Region const *>> smap;
        for (int i = 0; i < n; ++i) {
          auto sit0 =
              citerator(smap, order[i].x0, 1, &rmap.find(order[i].x0)->second);
          auto sit1 =
              citerator(smap, order[i].x1, 1, &rmap.find(order[i].x1)->second);
          std::vector<RegionMap::Region const *> pSplitRegions,
              pMergeRegions{&rmap.find(order[i].x2)->second};
          append(pSplitRegions, sit0->second, sit1->second);
          if (bcLabels[i] == BC_LABEL_SPLIT) // Already split - skip
          {
            splice(smap[order[i].x2], pSplitRegions);
          } else { // Merge - check
            double optSplitF1, optSplitPrec, optSplitRec;
            stats::pairF1<BigInt>(optSplitF1, optSplitPrec, optSplitRec,
                                  pSplitRegions, groundtruth, {BG_VAL});
            if (mergeF1s[i] > optSplitF1) {
              splice(smap[order[i].x2], pMergeRegions);
            } else {
              bcLabels[i] = BC_LABEL_SPLIT;
              splice(smap[order[i].x2], pSplitRegions);
            }
          }
        }
      }
    } else { // Use traditional Rand index
      parfor(
          0, n, true,
          [&bcLabels, &rmap, &order, &groundtruth](int i) {
            double mergeRI, splitRI;
            bcLabels[i] = genBoundaryClassificationLabelRI<BigInt>(
                mergeRI, splitRI, rmap, order[i].x0, order[i].x1, order[i].x2,
                groundtruth);
          },
          0);
    }
  } else { // Global optimal
    typedef TTree<NodeData> Tree;
    Tree tree;
    genTree(tree, order,
            [](Tree::Node &node, Label r) { node.data.label = r; });
    for (auto &node : tree) {
      if (node.isLeaf()) {
        node.data.bcLabel = BC_LABEL_MERGE;
        node.data.bestSplits.push_back(node.self);
      } else {
        std::vector<RegionMap::Region const *> pMergeRegions{
            &rmap.find(node.data.label)->second};
        std::vector<RegionMap::Region const *> pSplitRegions;
        for (auto c : node.children) {
          for (auto j : tree[c].data.bestSplits) {
            pSplitRegions.push_back(&rmap.find(tree[j].data.label)->second);
          }
        }
        double mergeF1 =
            stats::pairF1<BigInt>(pMergeRegions, groundtruth, {BG_VAL});
        double splitF1 =
            stats::pairF1<BigInt>(pSplitRegions, groundtruth, {BG_VAL});
        if (mergeF1 > splitF1) {
          node.data.bcLabel = BC_LABEL_MERGE;
          node.data.bestSplits.push_back(node.self);
        } else {
          node.data.bcLabel = BC_LABEL_SPLIT;
          for (auto c : node.children) {
            for (auto j : tree[c].data.bestSplits) {
              node.data.bestSplits.push_back(j);
            }
          }
        }
      }
    }
    if (globalOpt == 1) {
      // Enforce path consistency by merging
      std::queue<int> q;
      q.push(tree.root());
      while (!q.empty()) {
        int ni = q.front();
        q.pop();
        if (tree[ni].data.bcLabel == BC_LABEL_MERGE) {
          tree.traverseDescendants(
              ni, [](Tree::Node &node) { node.data.bcLabel = BC_LABEL_MERGE; });
        } else {
          for (auto c : tree[ni].children) {
            q.push(c);
          }
        }
      }
    } else if (globalOpt == 2) {
      // Enforce path consistency by splitting
      for (auto const &node : tree) {
        if (node.data.bcLabel == BC_LABEL_SPLIT) {
          tree.traverseAncestors(node.self, [](Tree::Node &tn) {
            tn.data.bcLabel = BC_LABEL_SPLIT;
          });
        }
      }
    } else {
      perr("Error: unsupported globalOpt type...");
    }
    // Collect bcLabels
    bcLabels.reserve(n);
    for (auto const &node : tree) {
      if (!node.isLeaf()) {
        bcLabels.push_back(node.data.bcLabel);
      }
    }
  }
  return bcLabels;
}


np::ndarray MyHmt::bc_label_ri_wrp(bp::list const &order,
                                   np::ndarray const &labels,
                                   np::ndarray const &groundtruth,
                                   bool const &usePairF1, int const &globalOpt,
                                   bool const &optSplit, bool const &tweak,
                                   double const &maxPrecDrop) {

  LabelImageType::Pointer segImage =
      nph::np_to_itk_label(bp::extract<np::ndarray>(labels));

  LabelImageType::Pointer truthImage =
      nph::np_to_itk_label(bp::extract<np::ndarray>(groundtruth));

  auto order_vec = nph::np_to_vector_triple<Label>(order);
  auto bcLabels =
      bc_label_ri_operation(order_vec, segImage, truthImage, usePairF1,
                            globalOpt, optSplit, tweak, maxPrecDrop);

  return nph::vector_to_np<int>(bcLabels);
}
