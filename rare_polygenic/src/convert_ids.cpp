#include <cstdint>
#include <vector>
#include <string>

namespace rvPRS
{

std::vector<std::int32_t> from_bytes(std::string &bytes)
{
    std::uint64_t length = bytes.size();
    std::vector<std::int32_t> ids(length / 3);
    std::int32_t parsed;
    unsigned int idx = 0;
    for (unsigned int pos = 0; pos < length; pos += 3)
    {
        parsed = *reinterpret_cast<const std::int32_t *>(&bytes[pos]);
        if (parsed & 0x00800000)
        {
            parsed |= 0xFF000000;
        }
        else
        {
            parsed &= 0x00FFFFFF;
        }
        ids[idx] = parsed;
        idx += 1;
    }
    return ids;
}

// Converts ints to byte sequences  e.g. [1000, 2000] ->  b'\xe8\x03\x00\xd0\x07\x00'
// where \xe8\x03\x00 is 1000 encoded in 3-bytes and \xd0\x07\x00 is 2000
// encoded in 3-bytes. This is requires less storage, particularly compared to
// storing the ints as comma-separated string of int values e.g. 
// '1000000,2000000' takes 15 bytes as string, but only 6 bytes when converted 
// to byte sequence.
std::string to_bytes(std::vector<std::int32_t> &ids)
{
    std::string bytes;
    bytes.resize(ids.size() * 3);
    std::uint32_t i = 0;
    for (auto &id: ids) {
        unsigned char *p = (unsigned char *)&id;
        bytes[i] = p[0];
        bytes[i + 1] = p[1];
        bytes[i + 2] = p[2];
        i += 3;
    }
    return bytes;
}

} // namespace