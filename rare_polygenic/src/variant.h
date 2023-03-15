#ifndef RVPRS_VARIANT_H_
#define RVPRS_VARIANT_H_

#include <algorithm>
#include <cstdint>
#include <vector>
#include <string>

#include "convert_ids.h"

namespace rvPRS {

class CppVariant
{
public:
  std::string varid;
  std::string rsid;
  std::string chrom;
  std::uint32_t pos;
  std::string ref;
  std::string alt;
  std::string symbol;
  std::string consequence;
  double missense_pathogenicity;
  double spliceai;
  double af;
  std::uint64_t ac;
  std::uint64_t an;
  double gnomad_af;
  double topmed_af;
  std::string _missing;
  std::string _homs;
  std::string _hets;
  std::vector<std::uint64_t> sample_idx;
  std::int32_t _hash;
  CppVariant(){};
  CppVariant(std::string varid, std::string rsid, std::string chrom,
             std::uint32_t pos, std::string ref, std::string alt,
             std::string symbol, std::string consequence,
             double missense_pathogenicity, double spliceai, double af,
             std::uint64_t ac, std::uint64_t an, double gnomad_af,
             double topmed_af, std::string _missing, std::string _homs,
             std::string _hets) : varid(varid), rsid(rsid), chrom(chrom),
                                  pos(pos), ref(ref), alt(alt),
                                  symbol(symbol), consequence(consequence),
                                  missense_pathogenicity(missense_pathogenicity),
                                  spliceai(spliceai), af(af), ac(ac), an(an),
                                  gnomad_af(gnomad_af), topmed_af(topmed_af),
                                  _missing(_missing), _homs(_homs),
                                  _hets(_hets){
                                    _hash = std::hash<std::string>{}(varid);
                                  };

  std::vector<std::int32_t> missing() { return from_bytes(_missing); }
  std::vector<std::int32_t> homs() { return from_bytes(_homs); }
  std::vector<std::int32_t> hets() { return from_bytes(_hets); }
  std::vector<std::int32_t> all_samples()
  {
    std::vector<std::int32_t> samples = homs();
    std::vector<std::int32_t> temp = hets();
    samples.insert(samples.end(), temp.begin(), temp.end());
    return samples;
  };
  void set_sample_idx(std::vector<std::uint64_t> &indices)
  {
    sample_idx = indices;
    std::sort(sample_idx.begin(), sample_idx.end());
  };
  void flip_alleles(std::vector<std::int32_t> &samples) {
    std::vector<std::int32_t> tmp_homs = homs();
    std::vector<std::int32_t> tmp_hets = hets();
    std::vector<std::int32_t> tmp_missing = missing();

    std::vector<std::int32_t> initial(tmp_homs.size() + tmp_hets.size());
    std::vector<std::int32_t> combined(tmp_homs.size() + tmp_hets.size() + tmp_missing.size());
    std::merge(tmp_homs.begin(), tmp_homs.end(), tmp_hets.begin(), tmp_hets.end(), initial.begin());
    std::merge(initial.begin(), initial.end(), tmp_missing.begin(), tmp_missing.end(), combined.begin());

    std::vector<std::int32_t> remainder;
    std::set_difference(samples.begin(), samples.end(),
                        combined.begin(), combined.end(),
                        std::inserter(remainder, remainder.begin()));

    _homs = to_bytes(remainder);
    ac = _hets.size() + 2 * _homs.size();
    af = (double)ac / (double)an;
  }
};

} // namespace rvPRS

#endif  // RVPRS_VARIANT_H_