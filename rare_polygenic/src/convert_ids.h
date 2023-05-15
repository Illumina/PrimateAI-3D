#ifndef RVPRS_CONVERT_IDS_H
#define RVPRS_CONVERT_IDS_H

#include <cstdint>
#include <vector>
#include <string>

namespace rvPRS {

std::vector<std::int32_t> from_bytes_cpp(std::string & bytes);
std::string to_bytes_cpp(std::vector<std::int32_t> &ids);

} // namespace

#endif