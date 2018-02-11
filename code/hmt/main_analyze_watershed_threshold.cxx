#include "glia_base.hxx"  // Label, f_true, f_null, perr
#include "glia_image.hxx" // LabelImage, RealImage

#include "type/big_num.hxx"        // BigInt
#include "type/point.hxx"          // Point
#include "type/tuple.hxx"          // TTriple
#include "type/region_map.hxx"     // TRegionMap
#include "type/boundary_table.hxx" // TBoundaryTable
#include "type/tree.hxx"           // TTree

#include "util/text_cmd.hxx"       // bpo::* and parse

#include "util/mp.hxx"             // parfor
#include "util/container.hxx"      // append, splice, citerator

#include "util/image_io.hxx"       // readImage
#include "util/image_alg.hxx"      // watershed
#include "util/image_stats.hxx"    // pairStats, pairF1
#include "util/image.hxx"          // transformImage

#include "util/stats.hxx"          // randIndex, precision, recall, f1, mean, (stddev)

#include "util/struct.hxx"         // genCountMap
#include "util/struct_merge.hxx"   // genMergeOrderGreedyUsingPbMean, genMergeOrderGreedyUsingPbApproxMedian, transformKeys

#include "hmt/tree_build.hxx"      // genTree
#include "hmt/bc_label.hxx"        // genBoundaryClassificationLabelF1, genBoundaryClassificationLabelRI, BC_LABEL_SPLIT, BC_LABEL_MERGE

#include <math.h>
#include <utility>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>

using namespace glia;
using namespace glia::stats;
using namespace glia::hmt;

template <typename TContainer>
double stddev (TContainer const& data, double mean)
{
  if (data.size() == 0) { return mean; }
  double ret = 0.0;
  for (auto const& x: data) {
    double dx = x - mean;
    ret += dx * dx;
  }
  return sqrt(ret / data.size());
}

// This function is the bulk of the operation function in main_pre_merge essentially verbatim
void pre_merge (LabelImage<DIMENSION>::Pointer& lblImage,
                const RealImage<DIMENSION>::Pointer& pbImage,
                const LabelImage<DIMENSION>::Pointer& mask,
                std::vector<int> const& sizeThresholds, double rpbThreshold)
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(lblImage, mask, false);
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  std::unordered_map<Label, double> rpbs;
  genMergeOrderGreedyUsingPbMean(order, saliencies, rmap, true, pbImage,
    [&pbImage, &sizeThresholds, &rmap, rpbThreshold, &rpbs]
    (TBoundaryTable<std::pair<double, int>, RegionMap> const& bt, TBoundaryTable<std::pair<double, int>, RegionMap>::iterator btit)
    {
      Label key0 = btit->first.first;
      Label key1 = btit->first.second;
      auto const* pr0 = &rmap.find(key0)->second;
      auto const* pr1 = &rmap.find(key1)->second;
      size_t sz0 = pr0->size(), sz1 = pr1->size();
      if (sz0 > sz1) {
        std::swap(key0, key1);
        std::swap(pr0, pr1);
        std::swap(sz0, sz1);
      }
      if (sz0 < sizeThresholds[0]) { return true; }
      if (sizeThresholds.size() > 1) {
        if (sz0 < sizeThresholds[1]) {
          auto rpit0 = rpbs.find(key0);
          if (rpit0 == rpbs.end()) {
            double rpb = 0.0;
            pr0->traverse([&pbImage, &rpb, rpbThreshold](RegionMap::Region::Point const& p) { rpb += pbImage->GetPixel(p); });
            rpb = sdivide(rpb, sz0, 0.0);
            rpbs[key0] = rpb;
            if (rpb > rpbThreshold) { return true; }
          } else if (rpit0->second > rpbThreshold) { return true; }
        }
        if (sz1 < sizeThresholds[1]) {
          auto rpit1 = rpbs.find(key1);
          if (rpit1 == rpbs.end()) {
            double rpb = 0.0;
            pr1->traverse([&pbImage, &rpb, rpbThreshold](RegionMap::Region::Point const& p) { rpb += pbImage->GetPixel(p); });
            rpb = sdivide(rpb, sz1, 0.0);
            rpbs[key1] = rpb;
            if (rpb > rpbThreshold) { return true; }
          } else if (rpit1->second > rpbThreshold) { return true; }
        }
      }
      return false;
    });
  std::unordered_map<Label, Label> lmap;
  transformKeys(lmap, order);
  transformImage(lblImage, rmap, lmap);
}

// This function is the bulk of the operation function in main_merge_order_pb essentially verbatim
void merge_order_pb (std::vector<TTriple<Label>>& order, std::vector<double>& saliencies,
                     const LabelImage<DIMENSION>::Pointer& lblImage,
                     const RealImage<DIMENSION>::Pointer& pbImage,
                     const LabelImage<DIMENSION>::Pointer& mask, int type)
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(lblImage, mask, true); // Only use contours
  if (type == 1) {
    genMergeOrderGreedyUsingPbApproxMedian(order, saliencies, rmap, false, pbImage,
         f_true<TBoundaryTable<std::vector<double>, RegionMap>&,
                TBoundaryTable<std::vector<double>, RegionMap>::iterator>,
         f_null<std::vector<double>&, Label, Label>);
  } else if (type == 2) {
    genMergeOrderGreedyUsingPbMean(order, saliencies, rmap, false, pbImage,
         f_true<TBoundaryTable<std::pair<double, int>, RegionMap>&,
                TBoundaryTable<std::pair<double, int>, RegionMap>::iterator>);
  } else { perr("Error: stats type must be 1 or 2"); }
}

// This is from main_bc_label_ri verbatim
struct NodeData {
  Label label;
  std::vector<int> bestSplits;
  int bcLabel = BC_LABEL_MERGE;
};

// This function is the bulk of the operation function in main_bc_label_ri essentially verbatim
void bc_label_ri (std::vector<int>& bcLabels,
                  const std::vector<TTriple<Label>>& order,
                  const LabelImage<DIMENSION>::Pointer& lblImage,
                  const LabelImage<DIMENSION>::Pointer& truthImage,
                  const LabelImage<DIMENSION>::Pointer& mask,
                  int globalOpt, bool usePairF1, bool optSplit, bool tweak, double maxPrecDrop)
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(lblImage, mask, order, false);
  int n = order.size();
  if (globalOpt == 0) {  // Local optimal
    bcLabels.resize(n);
    if (usePairF1) {     // Use pair F1
      std::vector<double> mergeF1s, splitF1s;
      mergeF1s.resize(n);
      splitF1s.resize(n);
      parfor(0, n, true, [&](int i) {
        bcLabels[i] = genBoundaryClassificationLabelF1<BigInt>(mergeF1s[i], splitF1s[i], rmap, order[i].x0, order[i].x1, order[i].x2, truthImage, tweak, maxPrecDrop);
      }, 0);
      if (optSplit) {
        std::unordered_map<Label, std::vector<RegionMap::Region const*>> smap;
        for (int i = 0; i < n; ++i) {
          auto sit0 = citerator(smap, order[i].x0, 1, &rmap.find(order[i].x0)->second);
          auto sit1 = citerator(smap, order[i].x1, 1, &rmap.find(order[i].x1)->second);
          std::vector<RegionMap::Region const*> pSplitRegions, pMergeRegions{&rmap.find(order[i].x2)->second};
          append(pSplitRegions, sit0->second, sit1->second);
          if (bcLabels[i] == BC_LABEL_SPLIT) { splice(smap[order[i].x2], pSplitRegions); } // Already split - skip
          else {  // Merge - check
            double optSplitF1, optSplitPrec, optSplitRec;
            stats::pairF1<BigInt>(optSplitF1, optSplitPrec, optSplitRec, pSplitRegions, truthImage, {BG_VAL});
            if (mergeF1s[i] > optSplitF1) { splice(smap[order[i].x2], pMergeRegions); }
            else {
              bcLabels[i] = BC_LABEL_SPLIT;
              splice(smap[order[i].x2], pSplitRegions);
            }
          }
        }
      }
    } else { // Use traditional Rand index
      parfor(0, n, true, [&](int i) {
        double mergeRI, splitRI;
        bcLabels[i] = genBoundaryClassificationLabelRI<BigInt>(mergeRI, splitRI, rmap, order[i].x0, order[i].x1, order[i].x2, truthImage);
      }, 0);
    }
  } else {  // Global optimal
    typedef TTree<NodeData> Tree;
    Tree tree;
    genTree(tree, order, [](Tree::Node& node, Label r) { node.data.label = r; });
    for (auto& node : tree) {
      if (node.isLeaf()) {
        node.data.bcLabel = BC_LABEL_MERGE;
        node.data.bestSplits.push_back(node.self);
      } else {
        std::vector<RegionMap::Region const*> pMergeRegions{&rmap.find(node.data.label)->second};
        std::vector<RegionMap::Region const*> pSplitRegions;
        for (auto c : node.children) {
          for (auto j : tree[c].data.bestSplits) {
            pSplitRegions.push_back(&rmap.find(tree[j].data.label)->second);
          }
        }
        double mergeF1 = stats::pairF1<BigInt>(pMergeRegions, truthImage, {BG_VAL});
        double splitF1 = stats::pairF1<BigInt>(pSplitRegions, truthImage, {BG_VAL});
        if (mergeF1 > splitF1) {
          node.data.bcLabel = BC_LABEL_MERGE;
          node.data.bestSplits.push_back(node.self);
        } else {
          node.data.bcLabel = BC_LABEL_SPLIT;
          for (auto c : node.children) {
            for (auto j : tree[c].data.bestSplits) { node.data.bestSplits.push_back(j); }
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
          tree.traverseDescendants(ni, [](Tree::Node& node) { node.data.bcLabel = BC_LABEL_MERGE; });
        } else { for (auto c : tree[ni].children) { q.push(c); } }
      }
    } else if (globalOpt == 2) {
      // Enforce path consistency by splitting
      for (auto const& node : tree) {
        if (node.data.bcLabel == BC_LABEL_SPLIT) {
          tree.traverseAncestors(node.self, [](Tree::Node& tn) { tn.data.bcLabel = BC_LABEL_SPLIT; });
        }
      }
    } else { perr("Error: unsupported globalOpt type"); }
    // Collect bcLabels
    bcLabels.reserve(n);
    for (auto const& node : tree) { if (!node.isLeaf()) { bcLabels.push_back(node.data.bcLabel); } }
  }
}

bool operation (std::vector<std::string> const& pbFiles,
                std::vector<std::string> const& truthFiles,
                std::vector<std::string> const& maskFiles,
                double level,
                std::vector<int> const& sizeThresholds, double rpbThreshold,
                int type,
                int globalOpt, bool usePairF1, bool optSplit, bool tweak, double maxPrecDrop)
{
  // Check arguments
  size_t n = pbFiles.size();
  if (n != truthFiles.size()) {
    perr("Error: pb and truth image sets must be the same length");
  }
  if (sizeThresholds.size() != 1 && sizeThresholds.size() != 2) {
    perr("Error: there must be 1 or 2 size thresholds");
  }
  
  // Calculate the regions and merge/split ratio (watershed + pre_merge + merge_order_pb + bc_label_ri)
  // Note: goes much faster with higher water levels [0.005 is 2.5x faster than 0.0 and 0.05 is 6x faster than that]
  std::vector<BigInt> tps(n), tns(n), fps(n), fns(n);
  std::vector<size_t> regionNums(n, 0);
  std::vector<size_t> merges(n, 0), splits(n, 0);
  for (size_t i = 0; i < n; ++i) {
    std::cout << std::setw(2) << i << "/" << std::setw(2) << n << "   " << std::setw(3) << (i*100)/n << "%" << std::endl;
      
    // Read in the images
    auto pb    = readImage<RealImage<DIMENSION>>(pbFiles[i]);
    auto truth = readImage<LabelImage<DIMENSION>>(truthFiles[i]);
    auto mask  = (i >= maskFiles.size() || maskFiles[i].empty()) ?
                    LabelImage<DIMENSION>::Pointer(nullptr) :
                    readImage<LabelImage<DIMENSION>>(maskFiles[i]);
    
    // Calculate the watershed
    auto label = watershed<LabelImage<DIMENSION>>(pb, level);
    
    // Run pre-merge (time-consuming step)
    pre_merge(label, pb, mask, sizeThresholds, rpbThreshold);
    
    // Calculate the confusion matrix for this image
    pairStats(tps[i], tns[i], fps[i], fns[i], label, truth, mask, {}, {BG_VAL});

    // Count the number of regions
    std::unordered_map<Label, size_t> cmap;
    genCountMap(cmap, label, mask);
    cmap.erase(BG_VAL);
    regionNums[i] = cmap.size();
    
    // Calculate merge/split ratio (time-consuming step)
    std::vector<TTriple<Label>> order;
    std::vector<double> saliencies;
    merge_order_pb(order, saliencies, label, pb, mask, type);
    std::vector<int> bcLabels;
    bc_label_ri(bcLabels, order, label, truth, mask,
                globalOpt, usePairF1, optSplit, tweak, maxPrecDrop);
    for (std::vector<int>::const_iterator j = bcLabels.begin(); j != bcLabels.end(); ++j) {
      if (*j == BC_LABEL_MERGE) { ++merges[i]; } else { ++splits[i]; }
    }
  }
  std::cout << std::setw(2) << n << "/" << std::setw(2) << n << "   100%" << std::endl;
  
  // Sum the values
  BigInt tp = 0, tn = 0, fp = 0, fn = 0;
  size_t nmerges = 0, nsplits = 0;
  for (size_t i = 0; i < n; ++i) {
    tp += tps[i]; tn += tns[i]; fp += fps[i]; fn += fns[i];
    nmerges += merges[i];
    nsplits += splits[i];
  }

  // Calculate and output basic statistics
  double prec, rec, f, ri;
  randIndex(ri, tp, tn, fp, fn);
  precision(prec, tp, fp);
  recall(rec, tp, fn);
  f1(f, prec, rec);
  double regionNumMean = mean(regionNums);
  double regionNumStd = stddev(regionNums, regionNumMean);
  std::cout << "Precision     = " << prec << std::endl;
  std::cout << "Recall        = " << rec << std::endl;
  std::cout << "F1            = " << f << std::endl;
  std::cout << "Rand Index    = " << ri << std::endl;
  std::cout << "Region # mean = " << regionNumMean << std::endl;
  std::cout << "Region # sd   = " << regionNumStd << std::endl;
  if (nsplits == 0) {
    std::cout << "Merge/split ratio = undefined" << std::endl;
  } else {
    std::cout << "Merge/split ratio = " << ((double)nmerges / nsplits) << std::endl;
  }
  
  return true;
}

int main (int argc, char* argv[])
{
  std::vector<std::string> pbFiles, truthFiles, maskFiles;
  std::vector<int> sizeThresholds;
  double level, rpbThreshold;
  int type = 1; // 1: median, 2: mean
  bool usePairF1 = true;
  bool optSplit = false;
  int globalOpt = 0;  // 0: Bypass, 1: RCC by merge, 2: RCC by split
  bool tweak = false;
  double maxPrecDrop = 1.0;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("pb,P", bpo::value<std::vector<std::string>>(&pbFiles)->required()->multitoken(), "Input pb image file name(s)")
      ("truth,T", bpo::value<std::vector<std::string>>(&truthFiles)->required()->multitoken(), "Input ground-truth image file name(s)")
      ("mask,m", bpo::value<std::vector<std::string>>(&maskFiles)->multitoken(), "Input mask image file name(s) (optional)")
      ("level,l", bpo::value<double>(&level)->required(), "Watershed water level")
      ("sizeThreshold,t", bpo::value<std::vector<int>>(&sizeThresholds)->required()->multitoken(), "Region size threshold(s)")
      ("rpbThreshold,b", bpo::value<double>(&rpbThreshold), "Region average boundary probability threshold")
      ("statsType,s", bpo::value<int>(&type), "Boundary intensity stats type (1: median, 2: mean) [default: 1]")
      ("f1", bpo::value<bool>(&usePairF1), "Whether to use pair F1 other than traditional Rand index [default: true]")
      ("opt,g", bpo::value<int>(&globalOpt), "Global optimal assignment type (0: none, 1: RCC by merge, 2: RCC by split) [default: 0]")
      ("optSplit,p", bpo::value<bool>(&optSplit), "Whether to determine based on optimal splits [default: false]")
      ("tweak,w", bpo::value<bool>(&tweak), "Whether to tweak conditions for thick boundaries [default: false]")
      ("mpd,d", bpo::value<double>(&maxPrecDrop), "Maximum precision drop allowed for merge [default: 1.0]");
  return parse(argc, argv, opts) &&
      operation(pbFiles, truthFiles, maskFiles, level, sizeThresholds, rpbThreshold,
                type, globalOpt, usePairF1, optSplit, tweak, maxPrecDrop) ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
