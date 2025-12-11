#include "catalog.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <array>

namespace {

// RAII mmap wrapper
struct MMap {
  int fd = -1;
  size_t len = 0;
  void* ptr = nullptr;

  static MMap map_file(const char* path) {
    MMap m;
    m.fd = ::open(path, O_RDONLY);
    if (m.fd < 0) throw std::runtime_error(std::string("open failed: ") + path);
    struct stat st;
    if (fstat(m.fd, &st) != 0) { ::close(m.fd); throw std::runtime_error("fstat failed"); }
    m.len = static_cast<size_t>(st.st_size);
    if (m.len == 0) { ::close(m.fd); throw std::runtime_error("empty file"); }
    m.ptr = ::mmap(nullptr, m.len, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.ptr == MAP_FAILED) { ::close(m.fd); throw std::runtime_error("mmap failed"); }
    return m;
  }

  ~MMap() {
    if (ptr && ptr != MAP_FAILED) ::munmap(ptr, len);
    if (fd >= 0) ::close(fd);
  }
};

// Compute the meta.bin layout exactly like ingest.py
struct MetaLayout {
  size_t sgnum, cell, density, elem_lo, elem_hi, flags, cif_off, cif_len, total;
};

MetaLayout layout_for(uint64_t N) {
  size_t off = 0;
  auto take = [&](size_t bytes){ size_t s=off; off+=bytes; return s; };
  MetaLayout L;
  L.sgnum   = take(N * sizeof(uint16_t));
  L.cell    = take(N * 6 * sizeof(float));
  L.density = take(N * sizeof(float));
  L.elem_lo = take(N * sizeof(uint64_t));
  L.elem_hi = take(N * sizeof(uint64_t));
  L.flags   = take(N * sizeof(uint8_t));
  L.cif_off = take(N * sizeof(uint64_t));
  L.cif_len = take(N * sizeof(uint64_t));
  L.total   = off;
  return L;
}

template <class T>
const T* ptr_at(const void* base, size_t off) {
  return reinterpret_cast<const T*>(reinterpret_cast<const unsigned char*>(base) + off);
}

} // namespace

Catalog Catalog::mmap_all(const char* meta_bin,
                          const char* ids_bin,
                          const char* cif_blob_path) {
  // mmap ids.bin first to learn N and id_bytes_len
  MMap ids_m = MMap::map_file(ids_bin);
  if (ids_m.len < 8) throw std::runtime_error("ids.bin too small");
  const uint32_t* hdr = static_cast<const uint32_t*>(ids_m.ptr);
  uint64_t N = hdr[0];
  uint32_t bytes_len = hdr[1];
  size_t ids_off_off = 8;
  size_t ids_off_bytes = static_cast<size_t>(N) * sizeof(uint32_t);
  size_t bytes_off = ids_off_off + ids_off_bytes;
  if (ids_m.len < bytes_off + bytes_len)
    throw std::runtime_error("ids.bin truncated");

  // mmap meta.bin and wire columns by computed layout
  MMap meta_m = MMap::map_file(meta_bin);
  MetaLayout L = layout_for(N);
  if (meta_m.len < L.total) throw std::runtime_error("meta.bin size mismatch");

  Catalog c;
  c.n = N;

  c.sgnum   = ptr_at<uint16_t>(meta_m.ptr, L.sgnum);
  c.cell    = ptr_at<float>(meta_m.ptr, L.cell);
  c.density = ptr_at<float>(meta_m.ptr, L.density);
  c.elem_lo = ptr_at<uint64_t>(meta_m.ptr, L.elem_lo);
  c.elem_hi = ptr_at<uint64_t>(meta_m.ptr, L.elem_hi);
  c.flags   = ptr_at<uint8_t>(meta_m.ptr,  L.flags);
  c.cif_off = ptr_at<uint64_t>(meta_m.ptr, L.cif_off);
  c.cif_len = ptr_at<uint64_t>(meta_m.ptr, L.cif_len);

  c.id_off        = ptr_at<uint32_t>(ids_m.ptr, ids_off_off);
  c.id_bytes      = reinterpret_cast<const char*>(ids_m.ptr) + bytes_off;
  c.id_bytes_len  = bytes_len;

  // mmap blob
  MMap blob_m = MMap::map_file(cif_blob_path);
  c.cif_blob = reinterpret_cast<const unsigned char*>(blob_m.ptr);
  c.cif_blob_size = blob_m.len;

  // Transfer ownership of mmaps to a leak-safe static; simple and robust for CLI use.
  // (Alternative: store in a pimpl to clean up deterministically on object destruction.)
  static std::vector<std::unique_ptr<MMap>> keep; // keeps mappings alive for process lifetime
  keep.emplace_back(new MMap(std::move(meta_m)));
  keep.emplace_back(new MMap(std::move(ids_m)));
  keep.emplace_back(new MMap(std::move(blob_m)));

  return c;
}

std::string_view Catalog::id_at(uint64_t i) const {
  if (i >= n) return {};
  uint32_t s = id_off[i];
  uint32_t e = (i+1 < n) ? id_off[i+1] : id_bytes_len;
  if (e <= s || e > id_bytes_len) return {};
  // strings are null-terminated; exclude the trailing '\0'
  const char* p = id_bytes + s;
  size_t len = e - s;
  if (len && p[len-1] == '\0') --len;
  return std::string_view(p, len);
}

