#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

struct Catalog {
  // row count (derived from ids.bin header)
  uint64_t n = 0;

  // Columns in meta.bin (read-only pointers into mmaps)
  const uint16_t* sgnum   = nullptr;   // N
  const float*    cell    = nullptr;   // N x 6 (row-major)
  const float*    density = nullptr;   // N
  const uint64_t* elem_lo = nullptr;   // N
  const uint64_t* elem_hi = nullptr;   // N
  const uint8_t*  flags   = nullptr;   // N
  const uint64_t* cif_off = nullptr;   // N
  const uint64_t* cif_len = nullptr;   // N

  // ids.bin
  const uint32_t* id_off   = nullptr;  // N offsets into id_bytes
  const char*     id_bytes = nullptr;  // bytes block (N null-terminated strings)
  uint32_t        id_bytes_len = 0;    // from ids.bin header

  // blob
  const unsigned char* cif_blob = nullptr;
  uint64_t             cif_blob_size = 0;

  // Construct from three files (POSIX mmap); throws on failure.
  static Catalog mmap_all(const char* meta_bin,
                          const char* ids_bin,
                          const char* cif_blob_path);

  // Lookups
  std::string_view id_at(uint64_t i) const;
  // (Optional hash index could go here later)
};

