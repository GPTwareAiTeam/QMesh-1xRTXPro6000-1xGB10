/* qmesh_expert_seam.cu: host-staged RC RDMA expert seam, see qmesh_expert_seam.h.
 *
 * Slab layout, identical on both peers, per layer L (stride bytes, 64B aligned):
 *   [req_off, +req_len)          request : x bf16[rows,5120] | ids i32[rows,6] | w bf16[rows,6]
 *   [resp_off, +resp_len)        response: out f32[rows,5120], or bf16/fp16 under the 16-bit wire
 *                                caps (protocol v5, SEAM_QES_OUT; see wire_caps)
 *   F_REQ  (u64, own 64B line)   request seq. Client: GPU publish kernel bumps it.
 *                                Server: remote RDMA flag WRITE lands here.
 *   F_RESP (u64, own 64B line)   response seq. Server: RDMA flag WRITE from its proxy.
 *                                Client: remote RDMA flag WRITE lands here.
 *
 * Protocol per call (seq = number of issues on that layer, starting at 1):
 *   issue  : GPU D2H copies the request into the slab; publish kernel then bumps
 *            F_REQ with st.release.sys (stream order guarantees the copies landed).
 *            Client proxy sees F_REQ advance, posts RDMA WRITE(payload) unsignaled,
 *            then RDMA WRITE(inline seq -> server F_REQ) signaled.
 *   server : proxy sees its F_REQ advance, computes the response into its slab,
 *            posts RDMA WRITE(payload), then RDMA WRITE(inline seq -> client F_RESP).
 *   collect: wait kernel spins (ld.acquire.sys) on F_RESP >= F_REQ, then the
 *            response is H2D copied to the caller's device buffer.
 *
 * Ordering: RC delivers in order on one QP, so a flag observed by the peer implies
 * its preceding payload is placed. Inline flag WRs copy the seq at post time, so no
 * source word is reused while in flight. A layer's buffers are reused only after the
 * caller collected, which the decode loop guarantees.
 *
 * Session model (protocol v2): the server owns the device, PD and one pinned slab+MR per row
 * bucket for its whole lifetime; each client session gets fresh QPs and CQs on those slabs, on
 * data ports assigned by the server. One control socket per session is held open for as long as
 * the session lives, so a client exit (clean, ctrl-c, kill -9) tears the session down and the next
 * client is served without a reload. Every failure path returns an error to the caller instead of
 * aborting: the server rejects one connection with a logged reason and keeps serving.
 *
 * Compute stub on the server (when no server_fn is installed): out[r][j] = float(x[r][j]) + (L+1),
 * rounded to the reply format.
 */
#include "qmesh_expert_seam.h"

#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

/* Row bucket ceiling. A bucket is one payload slot's width, so this is bounded by pinned host
 * memory (one slab per bucket on each side), not by the wire: 8192 rows is 240 MiB per side with
 * one slot (protocol v3 layout). */
constexpr int ROWS_MAX = 8192;
constexpr size_t LINE = 64;
/* How long a response may be outstanding before the slot is declared dead. 30 s is the right
 * default for a healthy seam, but a GX that is genuinely working - a first call that has to page a
 * cold layer in, a restack that runs inside the request - blows through it, and the client then calls
 * a live server "the seam transport is dead" and abandons the run. SEAM_ACK_TIMEOUT_S raises it for
 * such a run without changing what a hung transport looks like. */
uint64_t env_ack_timeout_ns() {
  const char* v = getenv("SEAM_ACK_TIMEOUT_S");
  if (!v || !*v) return 30ull * 1000 * 1000 * 1000;
  errno = 0;
  char* end = nullptr;
  unsigned long long s = strtoull(v, &end, 10);
  if (errno || end == v || (*end && *end != ' ') || s < 1 || s > 3600) {
    fprintf(stderr, "qes: SEAM_ACK_TIMEOUT_S=%s ignored (want whole seconds, 1..3600)\n", v);
    return 30ull * 1000 * 1000 * 1000;
  }
  return (uint64_t)s * 1000ull * 1000ull * 1000ull;
}
const uint64_t WAIT_TIMEOUT_NS = env_ack_timeout_ns();
/* How long a re-issue may wait for the server to acknowledge the slot's previous request before
 * the transport is declared stuck. The wait only happens when the caller has more requests in
 * flight than there are slots. */
constexpr uint64_t SLOT_ACK_TIMEOUT_NS = 30ull * 1000 * 1000 * 1000;

/* Control channel: negotiation hello/reply and the data-handshake guard words. */
constexpr uint32_t CTL_MAGIC = 0x32534551u;        /* "QES2" */
constexpr uint32_t CTL_REPLY_MAGIC = 0x32525351u;  /* "QSR2" */
constexpr uint32_t DATA_MAGIC = 0x33454851u;      /* "QHE3": protocol v3 slot layout */
constexpr uint32_t SLOT_MAGIC = 0x31544851u;       /* "QHT1": slot request header */
constexpr int CTL_HELLO_TIMEOUT_MS = 2000;        /* a silent TCP peer must not stall the server */
constexpr int DATA_HELLO_TIMEOUT_S = 5;          /* a dead data peer must not stall the accept loop */
constexpr int SOCK_IO_TIMEOUT_S = 15;
constexpr int CTL_REPLY_TIMEOUT_S = 60;   /* non-waiting client: the server answers at once */
constexpr int QUEUE_WAIT_S = 7200;        /* waiting client: a whole session lifetime */

constexpr uint32_t CTL_FLAG_WAIT = 1u;   /* client will queue behind the active session */
constexpr uint32_t CTL_FLAG_STATUS = 2u; /* status query: no session */

thread_local char g_err[512];

void set_err(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
  fprintf(stderr, "qes: %s\n", g_err);
  fflush(stderr);
}

/* Records the reason without printing it: for per-connection errors that the caller turns into a
 * SEAM_REJECT line with the detail attached, so the log does not read as if the server failed. */
void set_err_quiet(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

/* One line to stdout, shared with the Python server's own prints (single line, flushed). */
void say(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fputs(buf, stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string now_iso() {
  time_t t = time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char b[32];
  strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%S", &tmv);
  return b;
}

std::string csv(const int* v, int n) {
  std::string s;
  for (int i = 0; i < n; i++) {
    if (i) s += ',';
    s += std::to_string(v[i]);
  }
  return s;
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/* QES_TRACE=1 prints the wire geometry of every write. A remote-invalid-request completion on an
 * RDMA WRITE only says `opcode=1 port=<p> rows=<r>`: the responder rejected the address, the
 * length or the rkey, and nothing in the existing counters says which. These lines pin it to one
 * port by giving, on both sides and for the same port, the registered slab length, the stride, the
 * base address and the rkey, plus the offset and length of each posted write. */
bool qes_trace() {
  static const bool on = getenv("QES_TRACE") != nullptr;
  return on;
}

/* SEAM_HOOK_TIME=1 prints SEAM_HOOK_US for every served request: the microseconds the proxy spent
 * inside the compute hook. The hook is whichever pointer the server installed, so one switch
 * measures the Python cold path and the C hook the same way, and the difference between the two is
 * what the interpreter costs per call - no wire, no client queueing, no second instrument. */
bool qes_hook_time() {
  static const bool on = getenv("SEAM_HOOK_TIME") != nullptr;
  return on;
}

/* QES_TRACE=1 traces setup and the WRs each side posts; QES_TRACE=2 also traces every completion
 * and the bytes a compute produced, which tells "the peer never wrote it" apart from "the peer
 * wrote zeros". */
int trace_level() {
  static const int lvl = getenv("QES_TRACE") ? atoi(getenv("QES_TRACE")) : 0;
  return lvl;
}

const char* role_name(int role) {
  return role == QES_ROLE_SERVER ? "srv" : "cli";
}

constexpr uint64_t TRACE_FLAG = 1ULL << 63;

/* Wire caps (protocol v5) from this process's SEAM_QES_OUT: unset/f32 = 0 (the v4 fp32 reply, byte
 * for byte), bf16 = QES_CAP_OUT_BF16, fp16 = QES_CAP_OUT_FP16, anything else = -1, which makes
 * qes_server_open and qes_client_open refuse: a mistyped arm must not run on the previous wire.
 * Cached, because every handle's geometry must see one value for the process lifetime. */
int parse_caps() {
  const char* v = getenv("SEAM_QES_OUT");
  if (!v || !*v || !strcmp(v, "f32") || !strcmp(v, "fp32")) return 0;
  if (!strcmp(v, "bf16")) return QES_CAP_OUT_BF16;
  if (!strcmp(v, "fp16")) return QES_CAP_OUT_FP16;
  fprintf(stderr, "qes: SEAM_QES_OUT=%s is not f32|bf16|fp16: refusing to open\n", v);
  fflush(stderr);
  return -1;
}

int wire_caps() {
  static const int c = parse_caps();
  return c;
}

/* Bytes per reply element for a caps mask: 4 for fp32, 2 for either 16-bit format. */
int out_elem_of(int caps) { return (caps & (QES_CAP_OUT_BF16 | QES_CAP_OUT_FP16)) ? 2 : 4; }

const char* out_name(int caps) {
  return (caps & QES_CAP_OUT_BF16) ? "bf16" : (caps & QES_CAP_OUT_FP16) ? "fp16" : "f32";
}

/* The quiet NaN of the reply format, replicated into one 32-bit word, for poisoning a response
 * region that must not be read as an answer (bf16 0x7fc0, fp16 0x7e00, fp32 0x7fc00000). */
uint32_t nan_word(int caps) {
  if (caps & QES_CAP_OUT_BF16) return 0x7fc07fc0u;
  if (caps & QES_CAP_OUT_FP16) return 0x7e007e00u;
  return 0x7fc00000u;
}

/* Wire geometry for `rows` request rows, `slots` payload slots and a wire caps mask. One slot is
 *   [64B F_REQ line][64B F_RESP line][64B request header][x | ids | w][out]
 * and the request header (layer, rows) travels with the payload, so any slot serves any layer:
 * protocol v2 reserved a slot per layer, which multiplied the staging memory by QES_NL and capped
 * the usable row bucket at what 40 slabs allowed. The caps (protocol v5) decide the reply element
 * size; both sides compute this from the same mask, which the handshake requires to be equal. */
struct Geometry {
  size_t req_len, ids_off, w_off, resp_len, resp_off, stride, slab_bytes;
};

Geometry geometry(int rows, int slots, int caps) {
  Geometry g{};
  g.req_len = (size_t)rows * (QES_HIDDEN * 2 + QES_TOPK * 4 + QES_TOPK * 2);
  g.ids_off = (size_t)rows * QES_HIDDEN * 2;
  g.w_off = g.ids_off + (size_t)rows * QES_TOPK * 4;
  g.resp_len = (size_t)rows * QES_HIDDEN * out_elem_of(caps);
  g.resp_off = 3 * LINE + align_up(g.req_len, LINE);
  g.stride = g.resp_off + align_up(g.resp_len, LINE);
  g.slab_bytes = g.stride * (size_t)slots;
  return g;
}

/* Request header at the front of every request payload (protocol v3). */
struct SlotHdr {
  uint32_t magic; /* SLOT_MAGIC */
  uint32_t layer; /* QES_NL range */
  uint32_t rows;  /* rows this request really carries: 1..the bucket the handle was opened with.
                     The bucket sizes the slab, the wire geometry and the RDMA transfer; the compute
                     follows this field, so a 2560-row prefill chunk served by a 4096-row handle is a
                     2560-row compute on the GX and a 2560-row response. */
  uint32_t n_cold; /* client: cold experts the ids were clamped to, 0 = not supplied. The server
                      ignores it; the client's proxy checks the payload against it before sending
                      (see client_proxy), so an out-of-range id never reaches QES_ROLE_SERVER's
                      placement check as an unexplained fail-stop. */
  /* The seq this header was written for, i.e. the value publish_kernel will put in F_REQ. The
   * header is a host write and its payload copies are GPU work, so a slot that the client started
   * to reuse before the proxy had posted the previous request carried a *new* header over the old
   * payload: the proxy then checked the old ids against the new layer's cold set and refused a
   * request the caller never sent (2026-11-01, "id 268 past a cold count of 254"). Both proxies now
   * require this word to equal the seq they are handling, which turns that mix into a named
   * failure instead of a wrong answer. publish_kernel rewrites it on the device to the value it
   * puts in F_REQ, so a captured graph replays with a matching seq instead of the capture-time one
   * (a replayed layer publishes F_REQ = 2, 3, ... while its baked header would still say 1). */
  uint64_t seq;
  uint64_t pad2[5]; /* keep the header one 64 B line */
};

/* Data handshake, exchanged on a data port once per (session, bucket). */
struct Hello {
  uint32_t qpn, psn, lid, mtu;
  uint8_t gid[16];
  uint64_t base;
  uint32_t rkey;
  uint32_t layers;
  uint32_t rows;
  uint32_t magic;   /* DATA_MAGIC: catches a peer that speaks another data-handshake build */
  uint32_t session; /* control session id: rejects a stray connection from an older session */
  uint32_t port;    /* data port, for diagnostics */
  uint32_t slots;   /* payload slots per handle: the slab layout depends on it */
  uint32_t caps;    /* wire caps (protocol v5): the reply geometry depends on it. Takes the struct's
                       tail padding, so sizeof(Hello) is the v4 size. */
};
static_assert(sizeof(Hello) == 72, "data hello must keep its v4 size");

/* Control hello/reply: fixed size, little endian on both hosts. */
struct CtlHello {
  uint32_t magic;
  uint32_t version;
  uint32_t flags;
  uint32_t layers;
  uint32_t n_buckets;
  uint32_t buckets[QES_MAX_BUCKETS];
  uint32_t caps;        /* wire caps this client selected (protocol v5; was reserved[0]) */
  uint32_t reserved[1];
  char build[QES_BUILD_LEN];
};

struct CtlReply {
  uint32_t magic;
  uint32_t version;
  int32_t status;
  uint32_t session;
  uint32_t layers;
  uint32_t n_buckets;
  uint32_t n_requested;
  uint32_t buckets[QES_MAX_BUCKETS];
  uint32_t ports[QES_MAX_BUCKETS];
  uint32_t requested[QES_MAX_BUCKETS];
  uint32_t slots; /* payload slots per handle (protocol v3) */
  char build[QES_BUILD_LEN];
  char placement_sha[QES_BUILD_LEN];
  /* Four bytes taken off the message's tail for the server's wire caps (protocol v5), so the reply
   * keeps its v4 size and a peer of the other version reads a whole reply and refuses on `version`
   * instead of stalling on a short read. Writers bound the text by sizeof(message). */
  char message[QES_MSG_LEN - 4];
  uint32_t caps;
};
static_assert(sizeof(CtlHello) == 5 * 4 + QES_MAX_BUCKETS * 4 + 8 + QES_BUILD_LEN,
              "control hello must keep its v4 size");
static_assert(sizeof(CtlReply) == 7 * 4 + 3 * QES_MAX_BUCKETS * 4 + 4 + 2 * QES_BUILD_LEN +
                                      QES_MSG_LEN,
              "control reply must keep its v4 size");

/* ---- sockets --------------------------------------------------------------------------- */

void sock_opts(int fd, int rcv_s, int snd_s) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (rcv_s > 0) {
    timeval tv{rcv_s, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  if (snd_s > 0) {
    timeval tv{snd_s, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
}

/* Exact-length send/recv. False (with `why`) on peer close, timeout or error. */
bool xfer(int fd, void* buf, size_t n, bool send, const char** why) {
  char* p = (char*)buf;
  while (n) {
    ssize_t k = send ? ::send(fd, p, n, MSG_NOSIGNAL) : ::recv(fd, p, n, 0);
    if (k == 0) {
      *why = "peer closed";
      return false;
    }
    if (k < 0) {
      if (errno == EINTR) continue;
      *why = strerror(errno);
      return false;
    }
    p += k;
    n -= (size_t)k;
  }
  return true;
}

/* Same as xfer(recv), but the whole read must finish within timeout_ms: a peer that dribbles
 * its hello (or sends one byte and stops) must not hold the accept loop, which also watches the
 * live session's control fd. Used for the control hello only. */
bool xfer_deadline(int fd, void* buf, size_t n, int timeout_ms, const char** why) {
  const int64_t t_end = now_ns() + (int64_t)timeout_ms * 1000000LL;
  char* p = (char*)buf;
  while (n) {
    int64_t left_ms = (t_end - now_ns()) / 1000000LL;
    if (left_ms <= 0) {
      *why = "hello not complete in time";
      return false;
    }
    pollfd pf{fd, POLLIN, 0};
    int pr = poll(&pf, 1, (int)std::min<int64_t>(left_ms, 1000));
    if (pr < 0) {
      if (errno == EINTR) continue;
      *why = strerror(errno);
      return false;
    }
    if (pr == 0) continue;
    ssize_t k = ::recv(fd, p, n, MSG_DONTWAIT);
    if (k == 0) {
      *why = "peer closed";
      return false;
    }
    if (k < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      *why = strerror(errno);
      return false;
    }
    p += k;
    n -= (size_t)k;
  }
  return true;
}

int tcp_listen(int port, const char** why) {
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  if (ls < 0) {
    *why = strerror(errno);
    return -1;
  }
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_ANY);
  a.sin_port = htons((uint16_t)port);
  if (bind(ls, (sockaddr*)&a, sizeof(a)) || listen(ls, 8)) {
    *why = strerror(errno);
    close(ls);
    return -1;
  }
  return ls;
}

/* Connects with retries so a client may race the server's listener start-up. */
int tcp_connect_wait(const char* ip, int port, int timeout_ms, const char** why) {
  const int64_t t_end = now_ns() + (int64_t)timeout_ms * 1000000;
  /* ECONNREFUSED means the host answered and nothing is listening: worth a few seconds in case the
   * server is between bind and accept, but never worth the whole budget - a dead server should be
   * reported, not waited out. Anything else (blackhole, unreachable) keeps the full budget. */
  const int64_t t_refused = now_ns() + 5LL * 1000 * 1000 * 1000;
  const char* last = "no attempt";
  for (;;) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
      *why = strerror(errno);
      return -1;
    }
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) {
      *why = "bad peer address";
      close(s);
      return -1;
    }
    if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) return s;
    const int err = errno;
    last = strerror(err);
    close(s);
    const int64_t now = now_ns();
    if (now >= t_end || (err == ECONNREFUSED && now >= t_refused)) {
      *why = last;
      return -1;
    }
    usleep(100000);
  }
}

/* ---- device kernels ---- */

/* Bump the request seq. Single writer, GPU side. release.sys orders it after the
 * request copies for the CPU proxy.
 *
 * The header's seq field is written first, from the same value: a captured graph replays this kernel
 * but not the host code that filled the header, so a baked header would keep the capture-time seq
 * while F_REQ climbs on every replay and both proxies would refuse a correct request. `hdr_seq`
 * points at SlotHdr::seq in the slot's header line; the release store to F_REQ below is the doorbell,
 * so the field is visible to the proxy before it reads the payload. */
__global__ void publish_kernel(uint64_t* seq, uint64_t* hdr_seq) {
  uint64_t v = *seq + 1;
  asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(hdr_seq), "l"(v) : "memory");
  asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(seq), "l"(v) : "memory");
}

/* The slot header's constant fields (everything but `seq`, which publish_kernel owns). Written by a
 * 1-thread kernel from the values the issuing call was made with, so a captured graph's replay writes
 * the layer, row count and cold bound *it* was captured for. Staging them in host memory instead
 * cannot work: the fill runs at capture time only, and one line per layer is shared by all of that
 * layer's graphs. Kernel arguments are baked per graph, so there is nothing to share. */
__global__ void hdr_kernel(uint8_t* slot_hdr, uint32_t magic, uint32_t layer, uint32_t rows,
                           uint32_t n_cold) {
  SlotHdr* h = reinterpret_cast<SlotHdr*>(slot_hdr);
  h->magic = magic;
  h->layer = layer;
  h->rows = rows;
  h->n_cold = n_cold;
}

/* Spin until resp_seq >= req_seq. A dead transport must never trap the GPU: `__trap()` raised a
 * sticky `unspecified launch failure` that killed the whole client process for one lost handle.
 * After WAIT_TIMEOUT_NS the kernel raises the slot's fail word (bytes 8..11 of the freq line, which
 * nothing else uses), poisons the response slot with NaN and returns; the proxy sees that word,
 * marks the handle dead and prints why, and the next issue/collect on it raises in Python.
 *
 * The poisoning is what makes "loud" true for the call that timed out itself. Without it the collect
 * that gave up still copied the response slot, which holds the *previous* answer, and returned 0:
 * measured 2026-11-01 (failstop run 1101-094834) as `CALL_OK 3 finite=True` printed right after
 * "no response ... within 30 s". A dead seam must hand the caller NaN (isfinite() false), never
 * plausible rows.
 *
 * The same poisoning covers a *refused* request. A server that cannot answer writes a code into this
 * fail word and then releases the flag (server_proxy), so a waiter that wakes on the flag must read
 * the word once more: without that read the collect would copy the slot's previous answer and hand
 * the caller 8192 plausible rows computed for another layer, which is the one failure mode that
 * looks like success. The word is written before the flag on the same QP, so seeing the flag means
 * the word is there. */
__global__ void wait_kernel(const uint64_t* req_seq, const uint64_t* resp_seq, uint32_t* fail,
                            uint32_t* resp, int n_words, uint32_t nan_word, uint64_t timeout_ns) {
  uint64_t want = *req_seq;
  uint64_t t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (true) {
    uint64_t got;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(got) : "l"(resp_seq) : "memory");
    bool poison = false;
    if (got >= want) {
      uint32_t why;
      asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(why) : "l"(fail) : "memory");
      if (!why) return;
      poison = true;
    } else {
      uint64_t now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - t0 > timeout_ns) {
        asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(fail), "r"(1u) : "memory");
        poison = true;
      }
    }
    if (poison) {
      /* nan_word is the reply format's quiet NaN replicated over 32 bits (see nan_word()), so the
       * poisoned region reads NaN in fp32, bf16 or fp16 alike. */
      for (int i = threadIdx.x; i < n_words; i += blockDim.x) resp[i] = nan_word;
      return;
    }
  }
}

/* 16-bit reply -> the caller's fp32 rows, reading the mapped slab directly (no H2D staging). Two
 * elements per 32-bit load; __ldcv so nothing cached from the slot's previous reply is reused. The
 * widening is exact, so the fp32 the caller adds is bit for bit the bf16/fp16 the GX rounded to,
 * and a NaN-poisoned slot widens to NaN. Grid-stride, so any bucket fits a bounded grid. */
__global__ void widen_kernel(const uint32_t* __restrict__ src, float* __restrict__ dst, size_t n,
                             bool bf16) {
  const size_t n2 = (n + 1) / 2;
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n2;
       k += (size_t)gridDim.x * blockDim.x) {
    const uint32_t w = __ldcv(src + k);
    const uint16_t lo = (uint16_t)(w & 0xffffu), hi = (uint16_t)(w >> 16);
    float a, b;
    if (bf16) {
      a = __uint_as_float((uint32_t)lo << 16);
      b = __uint_as_float((uint32_t)hi << 16);
    } else {
      a = __half2float(__ushort_as_half(lo));
      b = __half2float(__ushort_as_half(hi));
    }
    dst[2 * k] = a;
    if (2 * k + 1 < n) dst[2 * k + 1] = b;
  }
}

}  // namespace

struct ServerSession;
/* Defined with ServerSession itself, below: the proxy threads run long before that. */
void session_note_reason(ServerSession* ss, const char* why);

struct qes_handle {
  int role = 0, rows = 0, dev = 0, port_idx = 1, gid_idx = 0, tcp_port = 0;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  ibv_port_attr port_attr{};
  uint8_t* slab = nullptr; /* pinned, mapped host */
  int slots = 1;           /* payload slots in this slab (in-flight requests per handle) */
  size_t slab_bytes = 0, stride = 0, req_len = 0, resp_len = 0, resp_off = 0;
  size_t ids_off = 0, w_off = 0;
  int caps = 0;            /* wire caps this handle's geometry was laid out for (protocol v5) */
  uint64_t remote_base = 0;
  uint32_t remote_rkey = 0;
  /* A server-side handle borrows the server's device, PD, slab and MR; only its CQ and QP are
   * owned (and destroyed) per session. */
  bool owns_ctx = true, owns_pd = true, owns_slab = true, owns_mr = true;
  std::atomic<bool> running{false};
  std::atomic<bool> dead{false};
  std::atomic<uint64_t> calls{0};
  std::thread proxy;
  uint64_t handled[QES_MAX_SLOTS] = {}; /* proxy-owned: last seq forwarded/answered per slot */
  /* Client-owned: the seq this client last wrote into each slot. The reuse rule used the GPU-written
   * F_REQ, which reads 0 until publish_kernel has actually executed, so a slot could look unused
   * while its first request was still queued and be overwritten before the proxy ever forwarded it.
   * This counter is advanced by the issuing thread and is therefore never behind the bytes. */
  uint64_t issued[QES_MAX_SLOTS] = {};
  /* Client side: requests in flight, oldest first. One layer may have several (a prefill chunk
   * split into micro-batches), so collect() pairs with the oldest request of that layer. */
  struct Pending {
    int layer = -1, slot = -1;
  } pending[QES_MAX_SLOTS];
  int n_pending = 0, next_slot = 0;
  /* QES_DUMP_IDS: pinned mirror of the ids argument of the request being issued. The proxy dumps
   * the slab bytes it forwards; this is the same copy taken straight from the caller's tensor, so
   * a dump line that already shows an out-of-range id separates a client value bug (the tensor
   * was wrong, or the allocator had recycled it while the copy was still queued) from a transport
   * bug (the copy landed, the wire did not carry it). */
  int32_t* dump_src = nullptr;
  qes_server_fn server_fn = nullptr; /* optional compute hook (server role) */
  void* server_ctx = nullptr;
  struct qes_client_session* owner = nullptr; /* client handles are unlinked on destroy */
  /* Server handles: the session this handle serves, so a refusal can name itself in the session's
   * end line (SEAM_SESSION_END reason=compute-refused-code-N) instead of a generic qp-error. */
  ServerSession* sess = nullptr;
};

/* =========================================================================================
 * Client session
 * ========================================================================================= */

namespace {

void nego_from_reply(const CtlReply& r, qes_nego* out) {
  memset(out, 0, sizeof(*out));
  out->status = r.status;
  out->version = (int)r.version;
  out->slots = (int)r.slots;
  out->layers = (int)r.layers;
  out->n_buckets = (int)r.n_buckets;
  out->n_requested = (int)r.n_requested;
  for (uint32_t i = 0; i < r.n_buckets && i < QES_MAX_BUCKETS; i++) {
    out->buckets[i] = (int)r.buckets[i];
    out->ports[i] = (int)r.ports[i];
  }
  for (uint32_t i = 0; i < r.n_requested && i < QES_MAX_BUCKETS; i++)
    out->requested[i] = (int)r.requested[i];
  out->caps = (int)r.caps;
  memcpy(out->build, r.build, QES_BUILD_LEN);
  memcpy(out->placement_sha, r.placement_sha, QES_BUILD_LEN);
  memcpy(out->message, r.message, sizeof(r.message));
  out->build[QES_BUILD_LEN - 1] = 0;
  out->placement_sha[QES_BUILD_LEN - 1] = 0;
  out->message[sizeof(r.message) - 1] = 0;
}

void ctl_hello_init(CtlHello* hello, const char* build, const int* buckets, int n, int layers,
                    uint32_t flags) {
  memset(hello, 0, sizeof(*hello));
  hello->magic = CTL_MAGIC;
  hello->version = QES_PROTO_VERSION;
  hello->flags = flags;
  hello->layers = (uint32_t)layers;
  hello->caps = (uint32_t)wire_caps();
  hello->n_buckets = (uint32_t)n;
  for (int i = 0; i < n && i < QES_MAX_BUCKETS; i++) hello->buckets[i] = (uint32_t)buckets[i];
  if (build) {
    strncpy(hello->build, build, QES_BUILD_LEN - 1);
    hello->build[QES_BUILD_LEN - 1] = 0;
  }
}

}  // namespace

struct qes_client_session {
  int fd = -1;
  char host[64] = {};
  int ctl_port = 0;
  uint32_t session = 0;
  qes_nego nego{};
  std::mutex mu;
  std::vector<qes_handle*> handles;
};

namespace {

size_t off_slot(const qes_handle* h, int i) { return (size_t)i * h->stride; }
size_t off_freq(const qes_handle* h, int i) { return off_slot(h, i); }
size_t off_fresp(const qes_handle* h, int i) { return off_slot(h, i) + LINE; }
size_t off_hdr(const qes_handle* h, int i) { return off_slot(h, i) + 2 * LINE; }
size_t off_req(const qes_handle* h, int i) { return off_slot(h, i) + 3 * LINE; }
size_t off_resp(const qes_handle* h, int i) { return off_slot(h, i) + h->resp_off; }

uint64_t load_acq(const uint8_t* base, size_t off) {
  return __atomic_load_n((const uint64_t*)(base + off), __ATOMIC_ACQUIRE);
}

uint32_t load_acq32(const uint8_t* base, size_t off) {
  return __atomic_load_n((const uint32_t*)(base + off), __ATOMIC_ACQUIRE);
}

/* Bytes 8..11 of a slot's freq line: raised by wait_kernel when a response never came. Offset 0
 * holds the request seq and the rest of the 64 B line is unused, so no slot geometry changes. */
size_t off_fail(const qes_handle* h, int i) { return off_freq(h, i) + 8; }

/* Polls completions. A QP error marks the handle dead and stops its proxy instead of aborting:
 * the owning session then tears down and the server keeps serving. */
bool drain_cq(qes_handle* h) {
  ibv_wc wc[16];
  int n;
  while ((n = ibv_poll_cq(h->cq, 16, wc)) > 0) {
    for (int i = 0; i < n; i++)
      if (trace_level() >= 2)
        fprintf(stderr,
                "SEAM_TRACE_CQE role=%s port=%d rows=%d opcode=%d wr_id=0x%llx kind=%s "
                "byte_len=%u status=%s\n",
                role_name(h->role), h->tcp_port, h->rows, (int)wc[i].opcode,
                (unsigned long long)wc[i].wr_id, (wc[i].wr_id & TRACE_FLAG) ? "flag" : "payload",
                wc[i].byte_len, ibv_wc_status_str(wc[i].status));
    for (int i = 0; i < n; i++)
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr,
                "qes: wc error %s opcode=%d port=%d rows=%d vend_err=0x%x wr_id=0x%llx "
                "off=%zu kind=%s\n",
                ibv_wc_status_str(wc[i].status), (int)wc[i].opcode, h->tcp_port, h->rows,
                wc[i].vendor_err, (unsigned long long)wc[i].wr_id,
                (size_t)(wc[i].wr_id & ~TRACE_FLAG),
                (wc[i].wr_id & TRACE_FLAG) ? "flag" : "payload");
        fflush(stderr);
        h->dead.store(true);
        h->running.store(false);
        return false;
      }
  }
  if (n < 0) {
    set_err("ibv_poll_cq failed (port=%d rows=%d)", h->tcp_port, h->rows);
    h->dead.store(true);
    h->running.store(false);
    return false;
  }
  return true;
}

/* RDMA WRITE of `len` bytes from local slab offset to the same offset remotely. */
bool post_payload(qes_handle* h, size_t off, uint32_t len) {
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)(h->slab + off);
  sge.length = len;
  sge.lkey = h->mr->lkey;
  ibv_send_wr wr{};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr_id = (uint64_t)off;
  wr.wr.rdma.remote_addr = h->remote_base + off;
  wr.wr.rdma.rkey = h->remote_rkey;
  /* A payload WRITE is unsignaled: the flag WRITE that follows it is the completion that matters,
   * and RC ordering makes the payload land first. QES_TRACE=2 asks for its CQE anyway: "the flag
   * arrived and the payload did not" is the shape of a write that was dropped, and then the two
   * completions are the only evidence there is. */
  if (trace_level() >= 2) wr.send_flags = IBV_SEND_SIGNALED;
  if (qes_trace())
    fprintf(stderr,
            "SEAM_TRACE_PUT role=%s port=%d rows=%u off=%zu len=%u remote=0x%016llx "
            "rkey=0x%08x slab=%zu\n",
            role_name(h->role), h->tcp_port, h->rows, off, len,
            (unsigned long long)(h->remote_base + off), h->remote_rkey, h->slab_bytes);
  ibv_send_wr* bad = nullptr;
  const int rc = ibv_post_send(h->qp, &wr, &bad);
  if (rc) {
    /* The errno is the diagnosis: ENOMEM is a full SQ (the proxy posted more than the queue
     * drains), EINVAL an sge the driver refused (lkey or addr+len outside the registered slab),
     * and ENOTSUP an opcode the QP was not built for. Name the sge too, so a log says which. */
    fprintf(stderr,
            "qes: ibv_post_send payload failed (port=%d rows=%d off=%zu len=%u slab=%zu): %s "
            "(%d)\n",
            h->tcp_port, h->rows, off, len, h->slab_bytes, strerror(rc), rc);
    fflush(stderr);
    set_err("ibv_post_send payload failed (port=%d rows=%d off=%zu len=%u slab=%zu): %s",
            h->tcp_port, h->rows, off, len, h->slab_bytes, strerror(rc));
    return false;
  }
  return true;
}

/* Inline (copied at post time) RDMA WRITE of an 8-byte seq to a remote flag word. */
bool post_flag(qes_handle* h, size_t remote_off, uint64_t seq) {
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)&seq;
  sge.length = 8;
  ibv_send_wr wr{};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr_id = (uint64_t)remote_off | TRACE_FLAG;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr = h->remote_base + remote_off;
  wr.wr.rdma.rkey = h->remote_rkey;
  if (qes_trace())
    fprintf(stderr,
            "SEAM_TRACE_FLAG role=%s port=%d rows=%u off=%zu seq=%llu remote=0x%016llx "
            "rkey=0x%08x slab=%zu\n",
            role_name(h->role), h->tcp_port, h->rows, remote_off, (unsigned long long)seq,
            (unsigned long long)(h->remote_base + remote_off), h->remote_rkey, h->slab_bytes);
  ibv_send_wr* bad = nullptr;
  const int rc = ibv_post_send(h->qp, &wr, &bad);
  if (rc) {
    fprintf(stderr, "qes: ibv_post_send flag failed (port=%d rows=%d off=%zu seq=%llu): %s (%d)\n",
            h->tcp_port, h->rows, remote_off, (unsigned long long)seq, strerror(rc), rc);
    fflush(stderr);
    set_err("ibv_post_send flag failed (port=%d rows=%d off=%zu seq=%llu): %s", h->tcp_port,
            h->rows, remote_off, (unsigned long long)seq, strerror(rc));
    return false;
  }
  return true;
}

/* Inline RDMA WRITE of one 4-byte word into the peer's slab (the server writes a refusal code into
 * the client's slot fail word with it). Same QP as post_flag, so a code posted here is on the wire
 * before a flag posted after it. */
bool post_u32(qes_handle* h, size_t remote_off, uint32_t v) {
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)&v;
  sge.length = 4;
  ibv_send_wr wr{};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr_id = (uint64_t)remote_off | TRACE_FLAG;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr = h->remote_base + remote_off;
  wr.wr.rdma.rkey = h->remote_rkey;
  ibv_send_wr* bad = nullptr;
  const int rc = ibv_post_send(h->qp, &wr, &bad);
  if (rc) {
    fprintf(stderr, "qes: ibv_post_send word off=%zu v=%u failed: %s (%d)\n", remote_off, v,
            strerror(rc), rc);
    fflush(stderr);
    return false;
  }
  return true;
}

/* The no-hook server path (stub benches): out[r][j] = float(x[r][j]) + (layer + 1) over the request's
 * own `rows` rows, not the bucket's, written in the reply format (round to nearest even under the
 * 16-bit caps), so a stub check proves the slot's geometry and the format together. */
void server_compute(qes_handle* h, int slot, int L, int rows) {
  const float add = (float)(L + 1);
  const uint16_t* x = (const uint16_t*)(h->slab + off_req(h, slot));
  uint8_t* out = h->slab + off_resp(h, slot);
  const int n = rows * QES_HIDDEN;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < n; i++) {
    uint32_t bits = (uint32_t)x[i] << 16;
    float f;
    memcpy(&f, &bits, 4);
    f += add;
    if (h->caps & QES_CAP_OUT_BF16)
      ((__nv_bfloat16*)out)[i] = __float2bfloat16_rn(f);
    else if (h->caps & QES_CAP_OUT_FP16)
      ((__half*)out)[i] = __float2half_rn(f);
    else
      ((float*)out)[i] = f;
    if (i < 4) o[i] = f;
  }
  if (trace_level() >= 2)
    fprintf(stderr,
            "SEAM_TRACE_VAL role=srv port=%d slot=%d layer=%d out=[%.3f %.3f %.3f %.3f] "
            "out_off=%zu out=%s src=0x%llx\n",
            h->tcp_port, slot, L, o[0], o[1], o[2], o[3], off_resp(h, slot), out_name(h->caps),
            (unsigned long long)(h->remote_base + off_resp(h, slot)));
}

/* QES_DUMP_IDS only: the tuples this client has already forwarded. A refusal that matches a tuple
 * forwarded earlier for the same slot is a stale payload (the proxy read the slot before, or
 * instead of, this request's copy); a refusal that matches nothing is a value only the current
 * request could have produced, i.e. the caller's. Small calls only, and the callers gate the whole
 * thing on the dump, so the compute path never touches it. */
struct WirePast {
  uint32_t layer, slot;
  uint64_t seq;
  int n;
  int32_t ids[QES_TOPK * 2];
};
constexpr int WIRE_PAST = 16;
WirePast g_wire_past[WIRE_PAST];
int g_wire_past_n = 0;

void wire_past_record(qes_handle* h, int slot, uint64_t seq, const int32_t* ids, int n) {
  if (n > QES_TOPK * 2) return;
  WirePast& p = g_wire_past[g_wire_past_n++ % WIRE_PAST];
  const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, slot));
  p.layer = hdr->layer;
  p.slot = (uint32_t)slot;
  p.seq = seq;
  p.n = n;
  memcpy(p.ids, ids, (size_t)n * sizeof(int32_t));
}

/* Newest forwarded tuple for this slot other than `skip_seq`, or nullptr. `skip` walks past that
 * many hits. The request being checked is already in the ring when the check runs (dump_wire_ids
 * records it first), so without `skip_seq` every refusal matched itself n/n and reported
 * STALE_PAYLOAD. */
const WirePast* wire_past_find(int slot, int skip, uint64_t skip_seq) {
  for (int r = 0; r < g_wire_past_n && r < WIRE_PAST; r++) {
    const WirePast& p = g_wire_past[(g_wire_past_n - 1 - r) % WIRE_PAST];
    if ((int)p.slot != slot) continue;
    if (p.seq == skip_seq) continue;
    if (skip-- > 0) continue;
    return &p;
  }
  return nullptr;
}

/* QES_DUMP_IDS=1: print the bytes this proxy is about to forward, per request, for calls small
 * enough that the values are readable. The proxy is the last point where the wire payload exists
 * as the client's own memory, and it costs a host read of memory it already owns, so this can run
 * on a live client without changing anything it measures. It exists because the worker's audit
 * (SEAM_IDS) can only say what arrived: pairing the two shows whether an id that the worker
 * rejects was sent by the client or mangled on the way, which is the difference between a client
 * value bug and a transport bug. */
bool dump_ids() {
  static const bool on = std::getenv("QES_DUMP_IDS") && atoi(std::getenv("QES_DUMP_IDS")) != 0;
  return on;
}

void dump_wire_ids(qes_handle* h, int slot, uint64_t seq) {
  if (!dump_ids() || h->rows > 8) return;
  const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, slot));
  const int32_t* ids = (const int32_t*)(h->slab + off_req(h, slot) + h->ids_off);
  const int n = h->rows * QES_TOPK;
  const uint16_t* w = (const uint16_t*)(h->slab + off_req(h, slot) + h->w_off);
  const uint16_t* x = (const uint16_t*)(h->slab + off_req(h, slot));
  auto bf16f = [](uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
  };
  fprintf(stderr,
          "SEAM_WIRE_IDS port=%d rows=%u slot=%d seq=%llu layer=%u hdr_rows=%u n_cold=%u n=%d "
          "ids=[",
          h->tcp_port, h->rows, slot, (unsigned long long)seq, hdr->layer, hdr->rows,
          hdr->n_cold, n);
  for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? " " : "", ids[i]);
  fprintf(stderr, "] w=[");
  for (int i = 0; i < n; i++) fprintf(stderr, "%s%.6g", i ? " " : "", bf16f(w[i]));
  fprintf(stderr, "] x0=%.6g x1=%.6g", bf16f(x[0]), bf16f(x[1 % QES_HIDDEN]));
  /* The same ids taken from the caller's tensor at issue time (qes_issue): equal to the slab ids
   * above means the bytes the proxy forwards are the bytes the caller handed over. The weights and
   * the first hidden values ride in the same payload, so they say whether the request as a whole is
   * the router's output for this layer or a pad/stale slot (weight 0 where a pick should be). */
  if (h->dump_src) {
    const int32_t* src = h->dump_src + (size_t)slot * n;
    fprintf(stderr, " src=[");
    for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? " " : "", src[i]);
    fprintf(stderr, "]");
  }
  /* What this slot carried the last time it was forwarded. A refusal whose ids equal this tuple
   * is the transport re-reading an old payload; one that differs is the caller's own value. */
  const WirePast* prev = wire_past_find(slot, 0, seq);
  if (prev) {
    fprintf(stderr, " prev=[layer=%u seq=%llu ids=", prev->layer, (unsigned long long)prev->seq);
    for (int i = 0; i < prev->n; i++) fprintf(stderr, "%s%d", i ? " " : "", prev->ids[i]);
    fprintf(stderr, "]");
  }
  wire_past_record(h, slot, seq, ids, n);
  fprintf(stderr, "\n");
  fflush(stderr);
}

/* Client proxy: forwards every new request seq in any slot. The slot's header (layer, rows) is
 * part of the payload, so the proxy posts the header and the payload in one WRITE. */
/* The last point before an id leaves this process: the bytes the NIC is about to send, read on the
 * proxy thread once the GPU's release publish has made them visible. A check on the issue path would
 * have to sync the stream, and a sync hides the failure it is looking for (2026-11-01: the failing
 * runs stop reproducing as soon as anything reads the ids back there), so it lives here where the
 * compute path pays nothing.
 * QES_ROLE_SERVER checks the same thing against its own placement and aborts the *process* on a
 * mismatch, which costs a full cold-pack reload (~150 s). Catching it on the sending side keeps the
 * server alive and names the request that would have killed it. */
bool client_check_ids(qes_handle* h, int slot, uint64_t seq) {
  const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, slot));
  const int32_t n_cold = (int32_t)hdr->n_cold;
  if (n_cold <= 0) return true; /* the caller supplied no bound: nothing to check against */
  const int32_t* ids = (const int32_t*)(h->slab + off_req(h, slot) + h->ids_off);
  /* Only the rows this request carries are checked. The slot is laid out for the bucket, so when
   * rows < bucket the tail still holds the previous request's ids - real bytes from another
   * request, which can be outside this layer's cold set for reasons that have nothing to do with
   * the caller (measured 2026-11-01: three stale picks at bucket 6 killed a decode handle after
   * 120 good requests, once the request carried its own 2-5 rows instead of padding to the
   * bucket). Scanning h->rows is what the padded protocol could do; it is wrong the moment a
   * partial request exists. */
  const int n = (int)hdr->rows * QES_TOPK;
  int bad = 0, first_at = -1, first_id = 0;
  for (int k = 0; k < n; k++) {
    if (ids[k] >= -1 && ids[k] < n_cold) continue;
    if (!bad++) {
      first_at = k;
      first_id = ids[k];
    }
  }
  if (!bad) return true;
  fprintf(stderr,
          "qes: NOT SENT: client ids out of range on rows=%u bucket=%u port=%d layer=%u slot=%d "
          "seq=%llu handled=%llu fresp=%llu: %d of %d pick(s) outside [-1,%d), first at %d: %d. The "
          "caller handed the seam an id its own cold set cannot hold, so the server's placement "
          "for this layer would index out of range; the request is dropped and this handle is "
          "taken down instead of the server.\n",
          hdr->rows, h->rows, h->tcp_port, hdr->layer, slot, (unsigned long long)seq,
          (unsigned long long)h->handled[slot],
          (unsigned long long)load_acq(h->slab, off_fresp(h, slot)), bad, n, n_cold, first_at,
          first_id);
  if (dump_ids()) {
    /* Same bytes as a tuple this slot already forwarded -> the payload is stale, not the caller's
     * value; nothing matches -> the caller produced it and the id space (first/n_cold) is the
     * suspect. Prints the two closest candidates for a partial match. */
    for (int skip = 0; skip < 2; skip++) {
      const WirePast* p = wire_past_find(slot, skip, seq);
      if (!p) break;
      int same = 0;
      for (int i = 0; i < n && i < p->n; i++) same += ids[i] == p->ids[i];
      fprintf(stderr,
              "qes: id history slot=%d prev_layer=%u prev_seq=%llu matched=%d/%d ids=[", slot,
              p->layer, (unsigned long long)p->seq, same, n);
      for (int i = 0; i < p->n; i++) fprintf(stderr, "%s%d", i ? " " : "", p->ids[i]);
      fprintf(stderr, "]%s\n", same == n ? " STALE_PAYLOAD" : "");
    }
    fflush(stderr);
  }
  /* A dropped request must not be mistaken for a computed one, and the caller copies the response
   * as soon as the ack lands: poison the response and raise the fail word *before* releasing the
   * ack, or the GPU thread races the NaN loop and hands the caller the slot's stale (zero) bytes.
   * The handle is dead either way, so the next seam call raises through seam_dead_check. */
  uint32_t* resp = (uint32_t*)(h->slab + off_resp(h, slot));
  const uint32_t nw = nan_word(h->caps);
  for (size_t k = 0; k < h->resp_len / sizeof(uint32_t); k++) resp[k] = nw;
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_store_n((uint32_t*)(h->slab + off_fail(h, slot)), 2u, __ATOMIC_RELEASE);
  __atomic_store_n((uint64_t*)(h->slab + off_fresp(h, slot)), load_acq(h->slab, off_freq(h, slot)),
                   __ATOMIC_RELEASE);
  h->dead.store(true);
  h->running.store(false);
  return false;
}

/* A response that never came leaves the GPU waiter with no escape hatch, so wait_kernel raises the
 * slot's fail word instead of trapping. Pick it up here, print it once, and take the handle down
 * the same way a CQ error does: the session fails with a Python exception, not a sticky CUDA error.
 *
 * Two ways a request ends this way, and they are not the same news. Word 1 is a timeout: nobody
 * answered and the transport is gone. Words 4 and 5 are the peer saying it could not compute this
 * request - out of memory, or its compute raised - while the listener stayed up, so the right move
 * is to reopen the bucket (the client does that lazily) or wait for room, not to restart the GX.
 * Say which one it was; the server's own SEAM_COMPUTE_REFUSED line carries the numbers. */
void client_proxy(qes_handle* h) {
  while (h->running.load(std::memory_order_relaxed)) {
    if (!drain_cq(h)) break;
    for (int i = 0; i < h->slots; i++) {
      const uint32_t why = load_acq32(h->slab, off_fail(h, i));
      if (!why) continue;
      if (why >= 4) {
        const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, i));
        fprintf(stderr,
                "qes: the server refused layer=%u rows=%u on port=%d slot=%d (code=%u: %s). The "
                "answer is NaN and this handle is finished; the server itself stayed up and is "
                "accepting, so the next session needs no reload. See its SEAM_COMPUTE_REFUSED "
                "line for avail_mib/swapped_mib.\n",
                hdr->layer, hdr->rows, h->tcp_port, i, why,
                why == 4 ? "not enough memory on the peer" : "the peer's compute failed");
      } else {
        fprintf(stderr,
                "qes: no response for layer slot %d on rows=%u port=%d within %llu s "
                "(seq=%llu answered=%llu): the seam transport is dead\n",
                i, h->rows, h->tcp_port, (unsigned long long)(WAIT_TIMEOUT_NS / 1000000000ull),
                (unsigned long long)load_acq(h->slab, off_freq(h, i)),
                (unsigned long long)h->handled[i]);
      }
      fflush(stderr);
      h->dead.store(true);
      h->running.store(false);
      return;
    }
    for (int i = 0; i < h->slots; i++) {
      uint64_t r = load_acq(h->slab, off_freq(h, i));
      if (r > h->handled[i]) {
        /* The flag says "seq r is ready", the header says which layer/rows/n_cold that seq is. They
         * are written by different engines (GPU kernel, caller's thread), so a slot that was reused
         * under a request that had not been forwarded yet shows one request's ids next to another's
         * layer. That is the only way a caller's own translate can look wrong here, and forwarding
         * it would compute one layer with another's cold ids: drop it and name the mix. */
        const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, i));
        /* rows is the request's real row count, bounded by the handle's bucket: it is not an equality
         * test any more (see SlotHdr::rows), but a header that claims more rows than the slab was
         * opened for, or none at all, is still a header from another world. */
        if (hdr->magic != SLOT_MAGIC || hdr->rows == 0 || hdr->rows > (uint32_t)h->rows ||
            hdr->seq != r) {
          fprintf(stderr,
                  "qes: NOT SENT: slot %d on rows=%u port=%d announced seq=%llu but its header says "
                  "magic=0x%08x layer=%u rows=%u seq=%llu n_cold=%u: header and payload in that slot "
                  "come from different requests, so the ids would be checked against another layer's "
                  "cold set. The request is dropped and this handle is taken down instead of the "
                  "server.\n",
                  i, h->rows, h->tcp_port, (unsigned long long)r, hdr->magic, hdr->layer, hdr->rows,
                  (unsigned long long)hdr->seq, hdr->n_cold);
          fflush(stderr);
          uint32_t* resp = (uint32_t*)(h->slab + off_resp(h, i));
          const uint32_t nw = nan_word(h->caps);
          for (size_t k = 0; k < h->resp_len / sizeof(uint32_t); k++) resp[k] = nw;
          __atomic_thread_fence(__ATOMIC_RELEASE);
          __atomic_store_n((uint32_t*)(h->slab + off_fail(h, i)), 3u, __ATOMIC_RELEASE);
          __atomic_store_n((uint64_t*)(h->slab + off_fresp(h, i)), r, __ATOMIC_RELEASE);
          h->dead.store(true);
          h->running.store(false);
          return;
        }
        dump_wire_ids(h, i, r);
        if (!client_check_ids(h, i, r)) return;
        /* Only the request's own rows go on the wire. The slot is laid out for the bucket, so x, ids
         * and w sit at bucket offsets and cannot be one WR for a partial request: the header and the
         * x prefix are contiguous, ids and w are posted separately. Same QP, same thread, and the
         * flag below is posted after them, so the peer sees the rows before it sees the seq (RC
         * completes in order), exactly as with the single bucket-sized WR this replaces. */
        const uint32_t rows = hdr->rows;
        const bool sent =
            post_payload(h, off_hdr(h, i), (uint32_t)(LINE + (size_t)rows * QES_HIDDEN * 2)) &&
            post_payload(h, off_req(h, i) + h->ids_off, rows * QES_TOPK * 4) &&
            post_payload(h, off_req(h, i) + h->w_off, rows * QES_TOPK * 2) &&
            post_flag(h, off_freq(h, i), r);
        if (!sent) {
          /* A WR the verbs layer rejects (SQ full, lkey or length outside this handle's registered
           * slab) leaves the request unsent and unretried, so this slot would spin its fail word
           * for WAIT_TIMEOUT_NS and every later issue would wait SLOT_ACK_TIMEOUT_NS for an ack
           * that can never come. Fail the handle here, like a CQ error: the next qes_issue or
           * qes_collect raises instead of stalling the GPU behind an ack gate. */
          h->dead.store(true);
          h->running.store(false);
          return;
        }
        /* Forwarded exactly once: `handled` is the last seq this proxy put on the wire, so the
         * next iteration cannot post the same slot again. Re-posting it until the SQ fills is how
         * one lost request became "ibv_post_send payload failed" on prod, measured 2026-11-01. */
        h->handled[i] = r;
        h->calls.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
}

/* Server proxy: computes and returns every new request seq in any slot. The layer comes from the
 * request header the client sent, so this loop is the same for every layer and several slots may
 * be computed/returned before the first response lands (the client keeps them in flight). */
void server_proxy(qes_handle* h) {
  while (h->running.load(std::memory_order_relaxed)) {
    if (!drain_cq(h)) break;
    for (int i = 0; i < h->slots; i++) {
      uint64_t r = load_acq(h->slab, off_freq(h, i));
      if (r <= h->handled[i]) continue;
      const SlotHdr* hdr = (const SlotHdr*)(h->slab + off_hdr(h, i));
      /* rows is the request's own row count (SlotHdr::rows): the bucket bounds it, and everything
       * below - the module call, the response - is sized by it, so the padding a big bucket leaves at
       * the end of a prefill chunk costs the GX nothing. */
      if (hdr->magic != SLOT_MAGIC || hdr->rows == 0 || hdr->rows > (uint32_t)h->rows ||
          hdr->layer >= (uint32_t)QES_NL || hdr->seq != r) {
        fprintf(stderr,
                "qes: bad request header slot=%d magic=0x%08x layer=%u rows=%u want_rows=%d "
                "seq=%llu want_seq=%llu\n",
                i, hdr->magic, hdr->layer, hdr->rows, h->rows, (unsigned long long)hdr->seq,
                (unsigned long long)r);
        fflush(stderr);
        h->dead.store(true);
        h->running.store(false);
        return;
      }
      const int L = (int)hdr->layer;
      const uint8_t* req = h->slab + off_req(h, i);
      const int rows = (int)hdr->rows;
      int refused = 0;
      if (h->server_fn) {
        /* SEAM_HOOK_TIME=1 times the call into the compute hook. Both paths come through this
         * line - the Python cold path and the C hook - so the difference between two runs is the
         * interpreter's cost on the server, not wire time and not client-side queueing: the hook
         * returns only after the answer is in the pinned slab, so this covers the whole forward
         * plus the copy out, blocked, exactly as the client waits for it. */
        const bool timed = qes_hook_time();
        const auto t0 = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        refused = h->server_fn(h->server_ctx, L, rows, req, (const int32_t*)(req + h->ids_off),
                               req + h->w_off, (float*)(h->slab + off_resp(h, i)));
        if (timed) {
          const double us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0).count();
          fprintf(stderr, "SEAM_HOOK_US port=%d layer=%d rows=%d us=%.1f\n", h->tcp_port, L, rows, us);
          fflush(stderr);
        }
      } else {
        server_compute(h, i, L, rows);
      }
      if (refused) {
        /* The hook could not answer: the box is out of memory, or its compute raised with the CUDA
         * context still usable. Post no response and kill nothing. The fail word tells the client's
         * GPU waiter to poison the slot (the caller sees NaN, never the slot's previous answer) and
         * its proxy to print why; ending this session only means the 113 GiB pack stays resident and
         * the next client's hello is accepted, instead of a 150-220 s reload for one request that
         * would not fit. The fail word is posted first: RC completes in order, so the client cannot
         * see the flag before the code that explains it. */
        fprintf(stderr,
                "qes: server refused layer=%d rows=%d slot=%d port=%d code=%d: ending this session, "
                "the server keeps accepting (see the hook's SEAM_COMPUTE_REFUSED line)\n",
                L, rows, i, h->tcp_port, refused);
        fflush(stderr);
        post_u32(h, off_fail(h, i), (uint32_t)refused);
        post_flag(h, off_fresp(h, i), r);
        /* Name it in the session's end line: the log then reads "refused code=4/5" and
         * "SEAM_SESSION_END reason=compute-refused-code-4/5" for the same event, which is what an
         * operator needs to tell a refusal apart from a wire fault. First reason wins. */
        if (h->sess) {
          char why[64];
          snprintf(why, sizeof(why), "compute-refused-code-%d", refused);
          session_note_reason(h->sess, why);
        }
        h->dead.store(true);
        h->running.store(false);
        return;
      }
      if (qes_trace())
        fprintf(stderr,
                "SEAM_TRACE_CMP role=srv port=%d rows=%u layer=%d slot=%d seq=%llu off_req=%zu "
                "off_resp=%zu req_len=%zu resp_len=%zu slab=%zu\n",
                h->tcp_port, h->rows, L, i, (unsigned long long)r, off_req(h, i), off_resp(h, i),
                h->req_len, h->resp_len, h->slab_bytes);
      /* Only the answered rows travel back: the client's collect copies the same count, so the
       * bucket's tail in either slab is never read (see qes_collect). The element size follows the
       * wire caps: half the bytes under a 16-bit reply. */
      if (!post_payload(h, off_resp(h, i),
                        (uint32_t)((size_t)rows * QES_HIDDEN * out_elem_of(h->caps))) ||
          !post_flag(h, off_fresp(h, i), r)) {
        /* One client's slot could not be answered: take this handle down so the session thread
         * ends it (SEAM_SESSION_END reason=qp-error) and the server goes back to accepting the
         * next client, instead of leaving a proxy that will never answer again. */
        h->dead.store(true);
        h->running.store(false);
        return;
      }
      h->handled[i] = r;
      h->calls.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

/* Fills the data handshake from the handle's QP/MR state. */
Hello make_hello(qes_handle* h, uint32_t session, uint32_t psn) {
  union ibv_gid lg{};
  if (ibv_query_gid(h->ctx, h->port_idx, h->gid_idx, &lg)) memset(lg.raw, 0, 16);
  Hello mine{};
  mine.qpn = h->qp->qp_num;
  mine.psn = psn;
  mine.lid = h->port_attr.lid;
  mine.mtu = h->port_attr.active_mtu;
  memcpy(mine.gid, lg.raw, 16);
  mine.base = (uint64_t)(uintptr_t)h->slab;
  mine.rkey = h->mr->rkey;
  mine.layers = QES_NL;
  mine.rows = (uint32_t)h->rows;
  mine.slots = (uint32_t)h->slots;
  mine.magic = DATA_MAGIC;
  mine.session = session;
  mine.port = (uint32_t)h->tcp_port;
  mine.caps = (uint32_t)h->caps;
  return mine;
}

/* Creates CQ+QP in INIT on an existing PD. */
bool qp_create(qes_handle* h, const char** why) {
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  h->cq = ibv_create_cq(h->ctx, 1024, nullptr, nullptr, 0);
  if (!h->cq) {
    *why = "ibv_create_cq";
    return false;
  }
  ibv_qp_init_attr qi{};
  qi.send_cq = h->cq;
  qi.recv_cq = h->cq;
  qi.qp_type = IBV_QPT_RC;
  qi.cap.max_send_wr = 512;
  qi.cap.max_recv_wr = 8;
  qi.cap.max_send_sge = 1;
  qi.cap.max_recv_sge = 1;
  qi.cap.max_inline_data = 64;
  h->qp = ibv_create_qp(h->pd, &qi);
  if (!h->qp) {
    *why = "ibv_create_qp";
    return false;
  }
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = (uint8_t)h->port_idx;
  a.qp_access_flags = access;
  if (ibv_modify_qp(h->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
    *why = "qp init";
    return false;
  }
  return true;
}
/* A state transition that fails is the one place the seam can lose a handle while both peers still
 * think the session is healthy, and the errno plus the state the hardware reports afterwards is all
 * the evidence there is: a responder QP that takes an incoming request while it is still in INIT or
 * RTR is moved to ERROR, so its own RTS transition then fails with EINVAL while the requester gets
 * "remote invalid request" on its first write. Print both instead of a bare reason string. */
void qp_fail(qes_handle* h, const char* step, int err, const Hello& peer, const char** why) {
  ibv_qp_attr a{};
  ibv_qp_init_attr i{};
  const char* state = "unknown";
  if (!ibv_query_qp(h->qp, &a, IBV_QP_STATE, &i)) {
    switch (a.qp_state) {
      case IBV_QPS_RESET: state = "reset"; break;
      case IBV_QPS_INIT: state = "init"; break;
      case IBV_QPS_RTR: state = "rtr"; break;
      case IBV_QPS_RTS: state = "rts"; break;
      case IBV_QPS_SQD: state = "sqd"; break;
      case IBV_QPS_SQE: state = "sqe"; break;
      case IBV_QPS_ERR: state = "error"; break;
      default: break;
    }
  }
  *why = step;
  fprintf(stderr,
          "qes: %s failed errno=%d (%s) state=%s qpn=%u peer_qpn=%u port=%d rows=%u slots=%d\n",
          step, err, strerror(err), state, h->qp->qp_num, peer.qpn, h->tcp_port, h->rows,
          h->slots);
}

/* RTR+RTS towards the peer's QP. */
bool qp_connect(qes_handle* h, const Hello& peer, uint32_t my_psn, const char** why) {
  ibv_qp_attr r{};
  r.qp_state = IBV_QPS_RTR;
  r.path_mtu = h->port_attr.active_mtu;
  r.dest_qp_num = peer.qpn;
  r.rq_psn = peer.psn;
  r.max_dest_rd_atomic = 4;
  r.min_rnr_timer = 12;
  r.ah_attr.is_global = 1;
  r.ah_attr.port_num = (uint8_t)h->port_idx;
  r.ah_attr.dlid = (uint16_t)peer.lid;
  r.ah_attr.sl = 0;
  r.ah_attr.src_path_bits = 0;
  memcpy(r.ah_attr.grh.dgid.raw, peer.gid, 16);
  r.ah_attr.grh.sgid_index = (uint8_t)h->gid_idx;
  r.ah_attr.grh.hop_limit = 1;
  if (ibv_modify_qp(h->qp, &r, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
    qp_fail(h, "qp rtr", errno, peer, why);
    return false;
  }
  ibv_qp_attr s{};
  s.qp_state = IBV_QPS_RTS;
  s.sq_psn = my_psn;
  s.timeout = 14;
  s.retry_cnt = 7;
  s.rnr_retry = 7;
  s.max_rd_atomic = 4;
  if (ibv_modify_qp(h->qp, &s, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                    IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)) {
    qp_fail(h, "qp rts", errno, peer, why);
    return false;
  }
  h->remote_base = peer.base;
  h->remote_rkey = peer.rkey;
  return true;
}

/* Checks the peer's data hello. */
bool check_hello(const Hello& peer, int rows, int slots, int port, uint32_t want_session, char* why,
                 size_t why_len) {
  if (peer.magic != DATA_MAGIC) {
    snprintf(why, why_len, "bad data hello magic=0x%08x hello_bytes=%zu", peer.magic, sizeof(Hello));
    return false;
  }
  if (peer.layers != QES_NL || peer.rows != (uint32_t)rows || peer.slots != (uint32_t)slots) {
    snprintf(why, why_len,
             "peer layout mismatch port=%d peer_layers=%u peer_rows=%u peer_slots=%u "
             "local_layers=%d local_rows=%u local_slots=%d hello_bytes=%zu",
             port, peer.layers, peer.rows, peer.slots, (int)QES_NL, (uint32_t)rows, slots,
             sizeof(Hello));
    return false;
  }
  if (peer.port != (uint32_t)port) {
    snprintf(why, why_len, "peer port mismatch port=%d peer_port=%u", port, peer.port);
    return false;
  }
  if (want_session && peer.session != want_session) {
    snprintf(why, why_len, "peer session mismatch sid=%u peer_sid=%u", want_session, peer.session);
    return false;
  }
  /* Each half laid out its reply region from its own caps: a session across two masks would write
   * the right bytes at the wrong width, which is plausible numbers rather than an error. */
  if (peer.caps != (uint32_t)wire_caps()) {
    snprintf(why, why_len, "peer caps mismatch port=%d peer_caps=0x%x (out=%s) local_caps=0x%x (out=%s)",
             port, peer.caps, out_name((int)peer.caps), (unsigned)wire_caps(), out_name(wire_caps()));
    return false;
  }
  return true;
}

/* Tears a handle down and unlinks it from its client session. */
void stop_handle(qes_handle* h) {
  if (!h) return;
  if (h->owner) {
    qes_client_session* s = h->owner;
    h->owner = nullptr;
    std::lock_guard<std::mutex> lk(s->mu);
    auto& v = s->handles;
    v.erase(std::remove(v.begin(), v.end(), h), v.end());
  }
  h->running.store(false);
  if (h->proxy.joinable()) h->proxy.join();
  if (h->qp) {
    ibv_destroy_qp(h->qp);
    h->qp = nullptr;
  }
  if (h->cq) {
    ibv_destroy_cq(h->cq);
    h->cq = nullptr;
  }
  if (h->owns_mr && h->mr) {
    ibv_dereg_mr(h->mr);
    h->mr = nullptr;
  }
  if (h->owns_slab && h->slab) {
    cudaFreeHost(h->slab);
    h->slab = nullptr;
  }
  if (h->dump_src) {
    cudaFreeHost(h->dump_src);
    h->dump_src = nullptr;
  }
  if (h->owns_pd && h->pd) {
    ibv_dealloc_pd(h->pd);
    h->pd = nullptr;
  }
  if (h->owns_ctx && h->ctx) {
    ibv_close_device(h->ctx);
    h->ctx = nullptr;
  }
  delete h;
}

/* Reads the server's negotiation reply. A client that asked to wait is told QES_ST_QUEUED at
 * once and then sits in the server's queue for as long as the current session runs, so the
 * deadline is ours here rather than the socket's 15 s SO_RCVTIMEO. The admission itself arrives
 * as a second reply (OK, or BUSY/reject if the queue entry died); only that one is returned. */
bool recv_reply(int fd, CtlReply* out, int wait, const char** why) {
  const int64_t t_end =
      now_ns() + (int64_t)(wait ? QUEUE_WAIT_S : CTL_REPLY_TIMEOUT_S) * 1000000000LL;
  for (;;) {
    CtlReply r{};
    char* p = (char*)&r;
    size_t n = sizeof(r);
    while (n) {
      pollfd pf{fd, POLLIN, 0};
      int pr = poll(&pf, 1, 1000);
      if (pr < 0) {
        if (errno == EINTR) continue;
        *why = strerror(errno);
        return false;
      }
      if (pr == 0) {
        if (now_ns() > t_end) {
          *why = wait ? "timed out waiting in the server queue"
                      : "timed out waiting for the negotiation reply";
          return false;
        }
        continue;
      }
      ssize_t k = ::recv(fd, p, n, MSG_DONTWAIT);
      if (k == 0) {
        *why = "peer closed";
        return false;
      }
      if (k < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        *why = strerror(errno);
        return false;
      }
      p += k;
      n -= (size_t)k;
    }
    if (r.magic != CTL_REPLY_MAGIC || r.status != QES_ST_QUEUED || !wait) {
      *out = r;
      return true;
    }
    fprintf(stderr, "qes: %s\n", r.message);
  }
}

}  // namespace

extern "C" const char* qes_last_error(void) { return g_err; }

extern "C" qes_client_session* qes_client_open(const char* peer_ip, int ctl_port,
                                               const char* build_id, const int* buckets,
                                               int n_buckets, int layers, int wait,
                                               qes_nego* out) {
  if (out) memset(out, 0, sizeof(*out));
  if (wire_caps() < 0) {
    set_err("qes_client_open: SEAM_QES_OUT=%s is not f32|bf16|fp16", getenv("SEAM_QES_OUT"));
    return nullptr;
  }
  if (!peer_ip || ctl_port <= 0) {
    set_err("qes_client_open: bad peer/control port (%s:%d)", peer_ip ? peer_ip : "(null)", ctl_port);
    return nullptr;
  }
  if (!buckets || n_buckets < 1 || n_buckets > QES_MAX_BUCKETS) {
    set_err("qes_client_open: requested %d buckets (1..%d)", n_buckets, QES_MAX_BUCKETS);
    return nullptr;
  }
  const char* why = "";
  int fd = tcp_connect_wait(peer_ip, ctl_port, 60000, &why);
  if (fd < 0) {
    set_err("qes_client_open: cannot reach seam control port %s:%d: %s", peer_ip, ctl_port, why);
    return nullptr;
  }
  sock_opts(fd, SOCK_IO_TIMEOUT_S, SOCK_IO_TIMEOUT_S);

  CtlHello hello;
  ctl_hello_init(&hello, build_id, buckets, n_buckets, layers, wait ? CTL_FLAG_WAIT : 0u);
  if (!xfer(fd, &hello, sizeof(hello), true, &why)) {
    set_err("qes_client_open: sending hello to %s:%d failed: %s", peer_ip, ctl_port, why);
    close(fd);
    return nullptr;
  }
  CtlReply reply{};
  if (!recv_reply(fd, &reply, wait, &why)) {
    set_err("qes_client_open: no negotiation reply from %s:%d: %s", peer_ip, ctl_port, why);
    close(fd);
    return nullptr;
  }
  if (reply.magic != CTL_REPLY_MAGIC) {
    set_err("qes_client_open: %s:%d is not a seam control port (magic=0x%08x, %zu bytes)", peer_ip,
            ctl_port, reply.magic, sizeof(reply));
    close(fd);
    return nullptr;
  }
  if (out) nego_from_reply(reply, out);
  if (reply.status != QES_ST_OK) {
    set_err("seam server %s:%d: %s", peer_ip, ctl_port, reply.message);
    close(fd);
    return nullptr;
  }
  if (reply.version != QES_PROTO_VERSION) {
    set_err("seam server %s:%d speaks protocol v%u, client is v%d: rebuild qmesh/native on both "
            "sides (server build %s)",
            peer_ip, ctl_port, reply.version, QES_PROTO_VERSION, reply.build);
    close(fd);
    return nullptr;
  }
  /* The server refuses a client whose caps differ from its own (handle_ctl_conn), so a granted
   * session already agrees; this is the same rule from this side, for a server that granted
   * anyway. The reply region's width follows the mask, so there is nothing safe to fall back to. */
  if (reply.caps != (uint32_t)wire_caps()) {
    set_err("seam server %s:%d runs wire caps=0x%x (out=%s) but this client's SEAM_QES_OUT selects "
            "0x%x (out=%s): run both halves with the same SEAM_QES_OUT",
            peer_ip, ctl_port, reply.caps, out_name((int)reply.caps), (unsigned)wire_caps(),
            out_name(wire_caps()));
    close(fd);
    return nullptr;
  }

  qes_client_session* s = new qes_client_session();
  s->fd = fd;
  s->ctl_port = ctl_port;
  s->session = reply.session;
  strncpy(s->host, peer_ip, sizeof(s->host) - 1);
  nego_from_reply(reply, &s->nego);
  return s;
}

extern "C" qes_handle* qes_client_handle(qes_client_session* s, const char* dev_name, int gid_idx,
                                          int bucket, int device_ordinal) {
  if (!s) {
    set_err("qes_client_handle: no session");
    return nullptr;
  }
  int idx = -1;
  for (int i = 0; i < s->nego.n_buckets; i++)
    if (s->nego.buckets[i] == bucket) idx = i;
  if (idx < 0) {
    set_err("qes_client_handle: bucket %d is not part of this session (server offers %s)", bucket,
            csv(s->nego.buckets, s->nego.n_buckets).c_str());
    return nullptr;
  }
  if (bucket < 1 || bucket > ROWS_MAX) {
    set_err("qes_client_handle: bucket %d out of range 1..%d", bucket, ROWS_MAX);
    return nullptr;
  }
  const int port = s->nego.ports[idx];
  const int slots = s->nego.slots;
  if (slots < 1 || slots > QES_MAX_SLOTS) {
    set_err("qes_client_handle: server advertises slots=%d, this build serves 1..%d", slots,
            QES_MAX_SLOTS);
    return nullptr;
  }
  /* The caps the control handshake agreed on decide the reply width: the same mask laid out the
   * server's slabs before any session existed. */
  const int caps = s->nego.caps;
  const Geometry g = geometry(bucket, slots, caps);

  qes_handle* h = new qes_handle();
  h->role = QES_ROLE_CLIENT;
  h->rows = bucket;
  h->dev = device_ordinal;
  h->gid_idx = gid_idx;
  h->tcp_port = port;
  h->port_idx = 1;
  h->slots = slots;
  h->caps = caps;
  h->req_len = g.req_len;
  h->ids_off = g.ids_off;
  h->w_off = g.w_off;
  h->resp_len = g.resp_len;
  h->resp_off = g.resp_off;
  h->stride = g.stride;
  h->slab_bytes = g.slab_bytes;

  const char* why = "";
  if (cudaSetDevice(h->dev) != cudaSuccess) {
    set_err("qes_client_handle: cudaSetDevice(%d) failed", h->dev);
    delete h;
    return nullptr;
  }
  if (cudaHostAlloc(&h->slab, h->slab_bytes, cudaHostAllocMapped | cudaHostAllocPortable) !=
      cudaSuccess) {
    set_err("qes_client_handle: cudaHostAlloc %zu bytes failed", h->slab_bytes);
    delete h;
    return nullptr;
  }
  memset(h->slab, 0, h->slab_bytes);
  if (dump_ids() && h->rows <= 8) {
    /* Small pinned mirror, one slice per slot, so the dump can show the ids argument itself next
     * to the slab bytes the proxy forwards (see qes_handle::dump_src). */
    if (cudaHostAlloc(&h->dump_src, (size_t)h->slots * h->rows * QES_TOPK * 4,
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
      h->dump_src = nullptr;
  }

  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  ibv_device* dev = nullptr;
  for (int i = 0; list && i < n; i++)
    if (strcmp(ibv_get_device_name(list[i]), dev_name) == 0) dev = list[i];
  if (!dev) {
    set_err("qes_client_handle: RDMA device %s not found", dev_name);
    if (list) ibv_free_device_list(list);
    stop_handle(h);
    return nullptr;
  }
  h->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!h->ctx) {
    set_err("qes_client_handle: ibv_open_device(%s) failed", dev_name);
    stop_handle(h);
    return nullptr;
  }
  if (ibv_query_port(h->ctx, h->port_idx, &h->port_attr)) {
    set_err("qes_client_handle: ibv_query_port failed");
    stop_handle(h);
    return nullptr;
  }
  h->pd = ibv_alloc_pd(h->ctx);
  if (!h->pd) {
    set_err("qes_client_handle: ibv_alloc_pd failed");
    stop_handle(h);
    return nullptr;
  }
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  h->mr = ibv_reg_mr(h->pd, h->slab, h->slab_bytes, access);
  if (!h->mr) {
    set_err("qes_client_handle: ibv_reg_mr(%zu bytes) failed", h->slab_bytes);
    stop_handle(h);
    return nullptr;
  }
  if (!qp_create(h, &why)) {
    set_err("qes_client_handle: %s failed (port=%d rows=%d)", why, port, bucket);
    stop_handle(h);
    return nullptr;
  }

  const uint32_t my_psn = (uint32_t)(lrand48() & 0xffffff);
  Hello mine = make_hello(h, s->session, my_psn);

  int fd = tcp_connect_wait(s->host, port, 60000, &why);
  if (fd < 0) {
    set_err("qes_client_handle: connect %s:%d (rows=%d) failed: %s", s->host, port, bucket, why);
    stop_handle(h);
    return nullptr;
  }
  sock_opts(fd, SOCK_IO_TIMEOUT_S, SOCK_IO_TIMEOUT_S);
  Hello peer{};
  bool ok = xfer(fd, &mine, sizeof(mine), true, &why);
  if (ok) ok = xfer(fd, &peer, sizeof(peer), false, &why);
  close(fd);
  if (!ok) {
    set_err("qes_client_handle: data handshake with %s:%d (rows=%d) failed: %s", s->host, port,
            bucket, why);
    stop_handle(h);
    return nullptr;
  }
  char why2[256];
  if (!check_hello(peer, bucket, slots, port, s->session, why2, sizeof(why2))) {
    set_err("qes_client_handle: rejected by %s:%d: %s", s->host, port, why2);
    stop_handle(h);
    return nullptr;
  }
  if (!qp_connect(h, peer, my_psn, &why)) {
    set_err("qes_client_handle: %s failed (port=%d rows=%d)", why, port, bucket);
    stop_handle(h);
    return nullptr;
  }
  /* One line per handle: what this side believes the peer's registered slab is. Pair it with the
   * server's SEAM_TRACE_EP line for the same port to catch a layout disagreement immediately. */
  if (qes_trace())
    fprintf(stderr,
            "SEAM_TRACE_HANDLE role=cli port=%d rows=%u slots=%d stride=%zu slab=%zu "
            "req=%zu resp=%zu remote_base=0x%016llx remote_rkey=0x%08x\n",
            port, bucket, slots, h->stride, h->slab_bytes, h->req_len, h->resp_len,
            (unsigned long long)h->remote_base, h->remote_rkey);
  /* The peer's hello may only be answered once this side is in RTS: the server sends its hello from
   * RTR+RTS (see session_handle_up), so "hello received" means "peer QP is ready for our writes". */
  if (qes_trace())
    fprintf(stderr, "SEAM_TRACE_RTS role=cli port=%d rows=%u qpn=%u\n", port, bucket,
            h->qp->qp_num);
  h->running.store(true);
  h->proxy = std::thread(client_proxy, h);
  {
    std::lock_guard<std::mutex> lk(s->mu);
    s->handles.push_back(h);
    h->owner = s;
  }
  return h;
}

extern "C" void qes_client_close(qes_client_session* s) {
  if (!s) return;
  for (;;) {
    qes_handle* h = nullptr;
    {
      std::lock_guard<std::mutex> lk(s->mu);
      if (s->handles.empty()) break;
      h = s->handles.front();
    }
    stop_handle(h);
  }
  if (s->fd >= 0) {
    const char bye = 'B';
    const char* why = "";
    xfer(s->fd, (void*)&bye, 1, true, &why);
    close(s->fd);
    s->fd = -1;
  }
  delete s;
}

/* =========================================================================================
 * Server
 * ========================================================================================= */

namespace {

/* One pinned slab + MR per row bucket, registered once and reused by every session. */
struct BucketEp {
  int rows = 0, port = 0;
  /* Listening socket for this bucket's data port, bound for the whole server lifetime. A
   * per-session listen() leaves a window in which the client - which is told every port in the
   * negotiation reply - connects to a later bucket before the server gets there and sees
   * ECONNREFUSED. Bound once, the connection queues in the backlog and the session that asks for
   * that bucket accepts it whenever it arrives. */
  int ls = -1;
  uint8_t* slab = nullptr;
  size_t slab_bytes = 0, stride = 0, req_len = 0, resp_len = 0, resp_off = 0, ids_off = 0,
         w_off = 0;
  ibv_mr* mr = nullptr;
};

struct Pending {
  int fd = -1;
  char ip[64] = {};
  CtlHello hello{};
};

}  // namespace

struct qes_server {
  char dev_name[64] = {};
  int gid_idx = 0, dev = 0, ctl_port = 0;
  int slots = 1; /* payload slots per handle, advertised to every client */
  int n_buckets = 0;
  int buckets[QES_MAX_BUCKETS] = {};
  char build[QES_BUILD_LEN] = {};
  char placement_sha[QES_BUILD_LEN] = {};
  bool allow_wait = false;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_port_attr port_attr{};
  qes_server_fn fn = nullptr;
  void* fn_ctx = nullptr;
  BucketEp eps[QES_MAX_BUCKETS];
  int listen_fd = -1;
  std::atomic<bool> stopping{false};
  uint32_t next_session = 1;
  struct ServerSession* active = nullptr;
  std::deque<Pending> waiting;
};

/* Session state. File scope, not anonymous-namespace: qes_server::active must point at this very
 * type, and a member declaration would otherwise inject a distinct ::ServerSession. */
struct ServerSession {
  qes_server* srv = nullptr;
  int fd = -1;
  char ip[64] = {};
  char build[QES_BUILD_LEN] = {};
  std::string since;
  uint32_t id = 0;
  int n = 0;
  int buckets[QES_MAX_BUCKETS] = {};
  int ports[QES_MAX_BUCKETS] = {};
  size_t bucket_idx[QES_MAX_BUCKETS] = {};
  std::atomic<bool> ending{false};
  std::atomic<bool> th_done{false};
  std::thread th;
  std::mutex mu;
  std::string reason; /* first end reason wins, set by the control thread */
  int64_t t0 = 0;
  std::vector<qes_handle*> handles; /* owned by the session thread once it starts */
};

/* First end reason wins: a refusal names itself, a later teardown must not overwrite it. */
void session_note_reason(ServerSession* ss, const char* why) {
  if (!ss) return;
  std::lock_guard<std::mutex> lk(ss->mu);
  if (ss->reason.empty()) ss->reason = why;
}


namespace {

/* The state of the occupied server, and the one hint that is useful to whoever reads it: a client
 * that is already queued gets this string inside its own "queued:" reply, so telling it to queue
 * there reads as if the server had not heard it. The hint belongs to the two sites that answer a
 * client which has not asked to wait -- status and refusal -- so it is appended there, not here. */
std::string busy_text(qes_server* s) {
  char buf[QES_MSG_LEN];
  if (!s->active)
    snprintf(buf, sizeof(buf), "server busy");
  else
    snprintf(buf, sizeof(buf), "server busy (session from %s since %s, sid=%u, %llus)",
             s->active->ip, s->active->since.c_str(), s->active->id,
             (unsigned long long)((now_ns() - s->active->t0) / 1000000000));
  return buf;
}

std::string busy_hint() { return "; start the client with SEAM_WAIT=1 to queue instead of failing"; }

bool send_reply(int fd, const CtlReply& r) {
  const char* why = "";
  bool ok = xfer(fd, (void*)&r, sizeof(r), true, &why);
  if (!ok) fprintf(stderr, "qes: control reply to fd=%d failed: %s\n", fd, why);
  return ok;
}

void reply_init(CtlReply* r, const qes_server* s) {
  memset(r, 0, sizeof(*r));
  r->magic = CTL_REPLY_MAGIC;
  r->version = QES_PROTO_VERSION;
  r->layers = QES_NL;
  r->slots = (uint32_t)s->slots;
  /* The caps this server's slabs were laid out for: every reply carries them, so a refused or a
   * status client can name the server's wire. */
  r->caps = (uint32_t)wire_caps();
  strncpy(r->build, s->build, QES_BUILD_LEN - 1);
  strncpy(r->placement_sha, s->placement_sha, QES_BUILD_LEN - 1);
}

void reject_reply(CtlReply* r, int status, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(r->message, sizeof(r->message), fmt, ap);
  va_end(ap);
  r->status = status;
}

/* Accepted buckets: the server's order, restricted to what the client asked for. */
int intersect(const qes_server* s, const CtlHello& hello, CtlReply* r) {
  int n = 0;
  for (int i = 0; i < s->n_buckets; i++)
    for (uint32_t j = 0; j < hello.n_buckets && j < QES_MAX_BUCKETS; j++)
      if ((int)hello.buckets[j] == s->buckets[i]) {
        r->buckets[n] = (uint32_t)s->buckets[i];
        r->ports[n] = (uint32_t)s->eps[i].port;
        n++;
        break;
      }
  r->n_buckets = (uint32_t)n;
  r->n_requested = std::min<uint32_t>(hello.n_buckets, QES_MAX_BUCKETS);
  for (uint32_t j = 0; j < r->n_requested; j++) r->requested[j] = hello.buckets[j];
  r->layers = QES_NL;
  r->slots = (uint32_t)s->slots;
  return n;
}

/* Brings up one bucket's data handle for `ss` from an already accepted data connection `fd`:
 * data hello, QP connect, proxy. The listener stays open for the server's lifetime (BucketEp::ls),
 * so a connection that arrives early simply waits in its backlog.
 *
 * `*stale` is set when the connection turns out not to belong to this session (dead hello or a
 * hello naming another session, port or row count). That is routine - the data ports are open for
 * the whole server lifetime - so the caller closes the connection and keeps accepting. */
qes_handle* session_handle_up(qes_server* s, ServerSession* ss, int k, int fd, bool* stale) {
  const BucketEp& ep = s->eps[ss->bucket_idx[k]];
  const int port = ep.port;
  const char* why = "";

  qes_handle* h = new qes_handle();
  h->role = QES_ROLE_SERVER;
  h->rows = ep.rows;
  h->dev = s->dev;
  h->gid_idx = s->gid_idx;
  h->tcp_port = port;
  h->port_idx = 1;
  h->ctx = s->ctx;
  h->pd = s->pd;
  h->port_attr = s->port_attr;
  h->slab = ep.slab;
  h->slots = s->slots;
  h->slab_bytes = ep.slab_bytes;
  h->stride = ep.stride;
  h->req_len = ep.req_len;
  h->resp_len = ep.resp_len;
  h->resp_off = ep.resp_off;
  h->ids_off = ep.ids_off;
  h->w_off = ep.w_off;
  h->caps = wire_caps();
  h->mr = ep.mr;
  h->owns_ctx = h->owns_pd = h->owns_slab = h->owns_mr = false;
  h->server_fn = s->fn;
  h->server_ctx = s->fn_ctx;
  h->sess = ss;

  /* Fresh sequence flags for this session: the slab is reused by the next client. */
  memset(h->slab, 0, h->slab_bytes);

  if (!qp_create(h, &why)) {
    set_err("seam session %u: %s failed (port=%d rows=%d)", ss->id, why, port, ep.rows);
    close(fd);
    stop_handle(h);
    return nullptr;
  }

  /* The data hello is the only thing separating this session's client from a connection left by an
   * earlier one, so bound the read tightly: a dead peer must not stall the accept loop. */
  sock_opts(fd, DATA_HELLO_TIMEOUT_S, DATA_HELLO_TIMEOUT_S);

  const uint32_t my_psn = (uint32_t)(lrand48() & 0xffffff);
  Hello mine = make_hello(h, ss->id, my_psn);
  Hello peer{};
  /* A rejected or unreadable hello closes the connection, so say why before we do. A bare close
   * leaves the peer in recv() until its own timeout, and that is how an unexpected client's knock
   * showed up as "data handshake failed: Resource temporarily unavailable" instead of as the real
   * difference: our hello carries our session id, rows, slots and port, so the peer's check_hello
   * turns the mismatch into one line ("peer session mismatch sid=... peer_sid=..."). Best effort -
   * the knocker may already be gone, and sending 96 bytes into a socket nobody reads cannot block
   * (the fd also carries SO_SNDTIMEO = DATA_HELLO_TIMEOUT_S). */
  auto say_goodbye = [&]() {
    const char* ignored = "";
    xfer(fd, &mine, sizeof(mine), true, &ignored);
  };
  /* Order matters here: read the client's hello, reach RTR+RTS, and only then answer with ours.
   * The client posts its first payload write as soon as our hello arrives and its own RTR+RTS is
   * done, so our QP must already be past INIT at that moment. A request that reaches a QP still in
   * INIT is answered with an "invalid request" NAK and kills a handle that was never used - that was
   * the decode-phase wc error, on exactly one freshly connected bucket per arm. Answering after
   * RTR+RTS makes the ordering explicit instead of a race: responder ready -> hello -> requester
   * ready -> first write. */
  bool ok = xfer(fd, &peer, sizeof(peer), false, &why);
  if (!ok) {
    set_err_quiet("seam session %u: data handshake on port %d (rows=%d) failed: %s", ss->id, port,
                  ep.rows, why);
    say_goodbye();
    close(fd);
    stop_handle(h);
    if (stale) *stale = true;
    return nullptr;
  }
  char why2[256];
  if (!check_hello(peer, ep.rows, s->slots, port, ss->id, why2, sizeof(why2))) {
    set_err_quiet("seam session %u from %s: rejected data connection on port %d: %s", ss->id,
                  ss->ip, port, why2);
    say_goodbye();
    close(fd);
    stop_handle(h);
    if (stale) *stale = true;
    return nullptr;
  }
  if (!qp_connect(h, peer, my_psn, &why)) {
    set_err("seam session %u: %s (port=%d rows=%d)", ss->id, why, port, ep.rows);
    say_goodbye();
    close(fd);
    stop_handle(h);
    return nullptr;
  }
  if (qes_trace())
    fprintf(stderr, "SEAM_TRACE_RTS role=srv port=%d rows=%u qpn=%u\n", port, ep.rows,
            h->qp->qp_num);
  ok = xfer(fd, &mine, sizeof(mine), true, &why);
  close(fd);
  if (!ok) {
    set_err_quiet("seam session %u: data handshake on port %d (rows=%d) failed: %s", ss->id, port,
                  ep.rows, why);
    stop_handle(h);
    if (stale) *stale = true;
    return nullptr;
  }
  h->running.store(true);
  h->proxy = std::thread(server_proxy, h);
  return h;
}

void session_teardown(ServerSession* ss) {
  for (auto* h : ss->handles) stop_handle(h);
  ss->handles.clear();
}

void session_thread(ServerSession* ss) {
  qes_server* s = ss->srv;
  std::string bucket_list = csv(ss->buckets, ss->n);
  std::string port_list = csv(ss->ports, ss->n);
  say("SEAM_SESSION_START ip=%s build=%s proto=%d buckets=%s ports=%s sid=%u caps=0x%x out=%s",
      ss->ip, ss->build, QES_PROTO_VERSION, bucket_list.c_str(), port_list.c_str(), ss->id,
      (unsigned)wire_caps(), out_name(wire_caps()));
  /* Data handles come up lazily: the client opens a bucket's handle on its first request of that
   * size, and may never touch some of the buckets it asked for (a decode client requests
   * 1,6,64 and only ever uses 6). So accept in *arrival* order, one connection per bucket, for as
   * long as the session lives - the control connection is what ends it, not a bucket that stays
   * quiet. Every granted bucket's listener is polled; the listening socket identifies the bucket
   * and the data hello re-checks session, port and rows.
   *
   * A connection that fails those checks belongs to someone else (an earlier session's reconnect,
   * a probe) and is closed and counted, not treated as a failed session: the data ports are open
   * for the whole server lifetime, so knockers are expected. */
  std::vector<char> up((size_t)ss->n, 0);
  int nup = 0, stale_total = 0;
  bool ok = true;
  while (nup < ss->n && !ss->ending.load()) {
    pollfd fds[QES_MAX_BUCKETS];
    int map[QES_MAX_BUCKETS];
    int nf = 0;
    for (int k = 0; k < ss->n; k++) {
      if (up[(size_t)k]) continue;
      const int ls = s->eps[ss->bucket_idx[k]].ls;
      if (ls < 0) continue;
      fds[nf] = {ls, POLLIN, 0};
      map[nf] = k;
      nf++;
    }
    if (nf == 0) {
      set_err("seam session %u: no data listener left for %d bucket(s)", ss->id, ss->n - nup);
      ok = false;
      break;
    }
    const int r = poll(fds, (nfds_t)nf, 250);
    if (r < 0) {
      if (errno == EINTR) continue;
      set_err("seam session %u: poll on data ports failed: %s", ss->id, strerror(errno));
      ok = false;
      break;
    }
    if (r == 0) continue;
    for (int i = 0; i < nf; i++) {
      if (!(fds[i].revents & POLLIN)) continue;
      sockaddr_in peer{};
      socklen_t plen = sizeof(peer);
      const int fd = accept(fds[i].fd, (sockaddr*)&peer, &plen);
      if (fd < 0) continue;
      const int k = map[i];
      bool stale = false;
      qes_handle* h = session_handle_up(s, ss, k, fd, &stale);
      if (!h) {
        if (!stale) {
          ok = false;
          break;
        }
        char ip[64] = "?";
        inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
        say("SEAM_REJECT ip=%s reason=stale-data-conn port=%d rows=%d sid=%u detail=%s", ip,
            s->eps[ss->bucket_idx[k]].port, s->eps[ss->bucket_idx[k]].rows, ss->id, g_err);
        /* Never fatal: the session is bounded by its control connection, so the only thing a flood
         * of strangers can do is keep this loop busy while the real client is idle - which is
         * exactly what it would be doing anyway. */
        stale_total++;
        continue;
      }
      ss->handles.push_back(h);
      up[(size_t)k] = 1;
      nup++;
    }
  }
  if (!ok) {
    say("SEAM_SESSION_REJECT ip=%s build=%s buckets=%s sid=%u reason=setup-failed", ss->ip,
        ss->build, bucket_list.c_str(), ss->id);
    session_teardown(ss);
    {
      std::lock_guard<std::mutex> lk(ss->mu);
      if (ss->reason.empty()) ss->reason = "setup-failed";
    }
    ss->ending.store(true);
    ss->th_done.store(true);
    return;
  }
  if (nup < ss->n)
    /* Not an error: the client used fewer buckets than it negotiated, or closed while a handle was
     * still being set up. Informational only - it must never tear the session down early. */
    say("SEAM_SESSION_PARTIAL ip=%s buckets=%s sid=%u up=%d of %d stale=%d", ss->ip,
        bucket_list.c_str(), ss->id, nup, ss->n, stale_total);


  while (!ss->ending.load()) {
    bool dead = false;
    for (auto* h : ss->handles)
      if (h->dead.load()) {
        dead = true;
        break;
      }
    if (dead) {
      std::lock_guard<std::mutex> lk(ss->mu);
      if (ss->reason.empty()) ss->reason = "qp-error";
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  uint64_t calls = 0;
  for (auto* h : ss->handles) calls += h->calls.load();
  std::string reason;
  {
    std::lock_guard<std::mutex> lk(ss->mu);
    if (ss->reason.empty()) ss->reason = "internal";
    reason = ss->reason;
  }
  say("SEAM_SESSION_END ip=%s build=%s buckets=%s sid=%u duration_s=%.1f calls=%llu reason=%s",
      ss->ip, ss->build, bucket_list.c_str(), ss->id, (now_ns() - ss->t0) / 1e9,
      (unsigned long long)calls, reason.c_str());
  session_teardown(ss);
  ss->th_done.store(true);
}

void end_active(qes_server* s, const char* reason);

bool start_session(qes_server* s, int fd, const char* ip, const CtlHello& hello);

void admit_waiters(qes_server* s) {
  while (!s->stopping.load() && !s->active && !s->waiting.empty()) {
    Pending p = std::move(s->waiting.front());
    s->waiting.pop_front();
    if (start_session(s, p.fd, p.ip, p.hello)) return;
    if (p.fd >= 0) close(p.fd);
  }
}

/* Drops queue entries whose client died while waiting (kill -9, Ctrl-C, a client that gave up):
 * the fd is readable only if the peer closed or reset it, since the protocol is server-driven
 * and a queued client sends nothing. Without this a dead waiter keeps its slot until the
 * running session ends, and a queue of corpses would reject a live client later on. */
void reap_waiters(qes_server* s) {
  if (s->waiting.empty()) return;
  size_t ndrop = 0;
  std::deque<Pending> keep;
  for (auto& p : s->waiting) {
    bool dead = p.fd < 0;
    if (!dead) {
      pollfd pf{p.fd, POLLIN, 0};
      int pr = poll(&pf, 1, 0);
      if (pr > 0 && (pf.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
        char b[1];
        ssize_t k = ::recv(p.fd, b, 1, MSG_DONTWAIT);
        if (k == 0) dead = true; /* closed while queued */
        else if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
          dead = true; /* reset */
      }
    }
    if (dead) {
      if (p.fd >= 0) close(p.fd);
      say("SEAM_QUEUE_DROP ip=%s reason=client-gone", p.ip);
      ndrop++;
    } else {
      keep.push_back(std::move(p));
    }
  }
  if (ndrop) s->waiting.swap(keep);
}

bool start_session(qes_server* s, int fd, const char* ip, const CtlHello& hello) {
  CtlReply r;
  reply_init(&r, s);
  if (intersect(s, hello, &r) <= 0) {
    std::string srv_buckets = csv(s->buckets, s->n_buckets);
    std::string cli_buckets =
        csv((const int*)hello.buckets, (int)std::min<uint32_t>(hello.n_buckets, QES_MAX_BUCKETS));
    reject_reply(&r, QES_ST_BADREQUEST, "no common row bucket: client asked %s, server serves %s",
                 cli_buckets.c_str(), srv_buckets.c_str());
    say("SEAM_REJECT ip=%s reason=no-common-bucket client=%s server=%s", ip, cli_buckets.c_str(),
        srv_buckets.c_str());
    send_reply(fd, r);
    close(fd);
    return false;
  }
  r.status = QES_ST_OK;
  const uint32_t sid = s->next_session++;
  r.session = sid;
  if (!send_reply(fd, r)) {
    close(fd);
    return false;
  }
  sock_opts(fd, 0, SOCK_IO_TIMEOUT_S);

  ServerSession* ss = new ServerSession();
  ss->srv = s;
  ss->fd = fd;
  strncpy(ss->ip, ip, sizeof(ss->ip) - 1);
  strncpy(ss->build, hello.build, QES_BUILD_LEN - 1);
  ss->build[QES_BUILD_LEN - 1] = 0;
  ss->since = now_iso();
  ss->id = sid;
  ss->t0 = now_ns();
  ss->n = (int)r.n_buckets;
  for (int k = 0; k < ss->n; k++) {
    ss->buckets[k] = (int)r.buckets[k];
    ss->ports[k] = (int)r.ports[k];
    for (int i = 0; i < s->n_buckets; i++)
      if (s->buckets[i] == ss->buckets[k]) ss->bucket_idx[k] = (size_t)i;
  }
  s->active = ss;
  ss->th = std::thread(session_thread, ss);
  return true;
}

void end_active(qes_server* s, const char* reason) {
  if (!s->active) return;
  ServerSession* ss = s->active;
  {
    std::lock_guard<std::mutex> lk(ss->mu);
    if (ss->reason.empty()) ss->reason = reason;
  }
  ss->ending.store(true);
  if (ss->th.joinable()) ss->th.join();
  if (ss->fd >= 0) close(ss->fd);
  delete ss;
  s->active = nullptr;
  admit_waiters(s);
}

/* A session whose control socket is already closed is over, even if the poll arm that reaps it has
 * not run yet. Accept and that arm fire in the same iteration and accept is handled first, so a
 * client that closes and immediately reconnects (every battery, every short run) used to be told
 * "server busy" about a session that had already said bye. Only a client that is still connected
 * keeps the server. After the hello the only byte a client writes on this socket is 'B' (bye). */
void reap_if_closed(qes_server* s) {
  if (!s->active || s->active->fd < 0) return;
  char b[1] = {0};
  ssize_t k = recv(s->active->fd, b, 1, MSG_PEEK | MSG_DONTWAIT);
  if (k == 0)
    end_active(s, "client-closed");
  else if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
    end_active(s, "client-reset");
  else if (k > 0 && b[0] == 'B')
    end_active(s, "client-bye");
}

/* Reads and answers one control connection. May start (or queue) a session. */
void handle_ctl_conn(qes_server* s, int fd, const char* ip) {
  sock_opts(fd, 2, SOCK_IO_TIMEOUT_S);

  CtlHello hello{};
  const char* why = "";
  if (!xfer_deadline(fd, &hello, sizeof(hello), CTL_HELLO_TIMEOUT_MS, &why)) {
    say("SEAM_REJECT ip=%s reason=control-hello-timeout (%s, limit %d ms); hello is %zu bytes",
        ip, why, CTL_HELLO_TIMEOUT_MS, sizeof(CtlHello));
    close(fd);
    return;
  }
  if (hello.magic != CTL_MAGIC) {
    CtlReply rep;
    reply_init(&rep, s);
    reject_reply(&rep, QES_ST_INCOMPAT,
                 "not a seam control hello (magic=0x%08x); client protocol v%d expected",
                 hello.magic, QES_PROTO_VERSION);
    say("SEAM_REJECT ip=%s reason=bad-control-magic magic=0x%08x", ip, hello.magic);
    send_reply(fd, rep);
    close(fd);
    return;
  }

  CtlReply rep;
  reply_init(&rep, s);
  if (hello.version != QES_PROTO_VERSION) {
    reject_reply(&rep, QES_ST_INCOMPAT,
                 "protocol mismatch: client v%u build=%.*s, server v%d build=%s; rebuild "
                 "qmesh/native on both sides",
                 hello.version, QES_BUILD_LEN, hello.build, QES_PROTO_VERSION, s->build);
    say("SEAM_REJECT ip=%s reason=protocol-mismatch client_v=%u server_v=%d client_build=%.*s", ip,
        hello.version, QES_PROTO_VERSION, QES_BUILD_LEN, hello.build);
    send_reply(fd, rep);
    close(fd);
    return;
  }
  if (hello.n_buckets < 1 || hello.n_buckets > QES_MAX_BUCKETS) {
    reject_reply(&rep, QES_ST_BADREQUEST, "malformed hello: n_buckets=%u (max %d)", hello.n_buckets,
                 QES_MAX_BUCKETS);
    say("SEAM_REJECT ip=%s reason=malformed-hello n_buckets=%u", ip, hello.n_buckets);
    send_reply(fd, rep);
    close(fd);
    return;
  }
  if (hello.layers != QES_NL) {
    reject_reply(&rep, QES_ST_INCOMPAT,
                 "layer count mismatch: client layers=%u, server layers=%d; the seam layout must "
                 "match on both sides",
                 hello.layers, QES_NL);
    say("SEAM_REJECT ip=%s reason=layer-mismatch client_layers=%u server_layers=%d", ip,
        hello.layers, QES_NL);
    send_reply(fd, rep);
    close(fd);
    return;
  }

  if (hello.flags & CTL_FLAG_STATUS) {
    rep.status = QES_ST_STATUS;
    rep.n_buckets = (uint32_t)s->n_buckets;
    rep.n_requested = std::min<uint32_t>(hello.n_buckets, QES_MAX_BUCKETS);
    for (int i = 0; i < s->n_buckets; i++) {
      rep.buckets[i] = (uint32_t)s->buckets[i];
      rep.ports[i] = (uint32_t)s->eps[i].port;
    }
    snprintf(rep.message, sizeof(rep.message), "%s%s", s->active ? busy_text(s).c_str() : "idle",
             s->active ? busy_hint().c_str() : "");
    send_reply(fd, rep);
    close(fd);
    return;
  }
  /* After the status probe (which opens no session and touches no payload, so a caps mismatch
   * cannot make it wrong) and before any session: the masks decide the reply width, and each half
   * laid its slabs out from its own before this connection existed. */
  if (hello.caps != (uint32_t)wire_caps()) {
    reject_reply(&rep, QES_ST_INCOMPAT,
                 "wire caps mismatch: client caps=0x%x (out=%s), server caps=0x%x (out=%s); run "
                 "both halves with the same SEAM_QES_OUT",
                 hello.caps, out_name((int)hello.caps), (unsigned)wire_caps(), out_name(wire_caps()));
    say("SEAM_REJECT ip=%s reason=caps-mismatch client_caps=0x%x server_caps=0x%x", ip, hello.caps,
        (unsigned)wire_caps());
    send_reply(fd, rep);
    close(fd);
    return;
  }

  reap_if_closed(s);
  if (s->active) {
    if ((hello.flags & CTL_FLAG_WAIT) && s->allow_wait &&
        s->waiting.size() < (size_t)QES_MAX_BUCKETS) {
      rep.status = QES_ST_QUEUED;
      snprintf(rep.message, sizeof(rep.message), "queued: %s (position %zu)", busy_text(s).c_str(),
               s->waiting.size() + 1);
      say("SEAM_QUEUE ip=%s waiters=%zu", ip, s->waiting.size() + 1);
      if (send_reply(fd, rep)) {
        Pending pd;
        pd.fd = fd;
        strncpy(pd.ip, ip, sizeof(pd.ip) - 1);
        pd.hello = hello;
        s->waiting.push_back(pd);
        return; /* fd stays open until admitted */
      }
      close(fd);
      return;
    }
    rep.status = QES_ST_BUSY;
    snprintf(rep.message, sizeof(rep.message), "%s%s", busy_text(s).c_str(), busy_hint().c_str());
    say("SEAM_REJECT ip=%s reason=busy active_sid=%u active_ip=%s", ip, s->active->id,
        s->active->ip);
    send_reply(fd, rep);
    close(fd);
    return;
  }

  if (!start_session(s, fd, ip, hello)) close(fd);
}

}  // namespace

extern "C" qes_server* qes_server_open(const char* dev_name, int gid_idx, int ctl_port,
                                       int device_ordinal, const int* buckets, int n_buckets,
                                       const char* build_id, const char* placement_sha,
                                       int allow_wait, int slots, qes_nego* err) {
  if (err) memset(err, 0, sizeof(*err));
  if (wire_caps() < 0) {
    set_err("qes_server_open: SEAM_QES_OUT=%s is not f32|bf16|fp16", getenv("SEAM_QES_OUT"));
    return nullptr;
  }
  if (!dev_name || ctl_port <= 0 || !buckets || n_buckets < 1 || n_buckets > QES_MAX_BUCKETS) {
    set_err("qes_server_open: bad arguments");
    return nullptr;
  }
  for (int i = 0; i < n_buckets; i++)
    if (buckets[i] < 1 || buckets[i] > ROWS_MAX) {
      set_err("qes_server_open: bucket %d out of range 1..%d", buckets[i], ROWS_MAX);
      return nullptr;
    }
  for (int i = 1; i < n_buckets; i++)
    if (buckets[i] <= buckets[i - 1]) {
      set_err("qes_server_open: buckets must be ascending and unique");
      return nullptr;
    }
  if (slots < 1 || slots > QES_MAX_SLOTS) {
    set_err("qes_server_open: slots must be 1..%d (got %d)", QES_MAX_SLOTS, slots);
    return nullptr;
  }

  qes_server* s = new qes_server();
  strncpy(s->dev_name, dev_name, sizeof(s->dev_name) - 1);
  s->gid_idx = gid_idx;
  s->dev = device_ordinal;
  s->ctl_port = ctl_port;
  s->n_buckets = n_buckets;
  for (int i = 0; i < n_buckets; i++) s->buckets[i] = buckets[i];
  if (build_id) strncpy(s->build, build_id, QES_BUILD_LEN - 1);
  if (placement_sha) strncpy(s->placement_sha, placement_sha, QES_BUILD_LEN - 1);
  s->allow_wait = allow_wait != 0;
  s->slots = slots;

  const char* why = "";
  if (cudaSetDevice(s->dev) != cudaSuccess) {
    set_err("qes_server_open: cudaSetDevice(%d) failed", s->dev);
    delete s;
    return nullptr;
  }
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  ibv_device* dev = nullptr;
  for (int i = 0; list && i < n; i++)
    if (strcmp(ibv_get_device_name(list[i]), s->dev_name) == 0) dev = list[i];
  if (!dev) {
    set_err("qes_server_open: RDMA device %s not found", s->dev_name);
    if (list) ibv_free_device_list(list);
    delete s;
    return nullptr;
  }
  s->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!s->ctx) {
    set_err("qes_server_open: ibv_open_device(%s) failed", s->dev_name);
    delete s;
    return nullptr;
  }
  if (ibv_query_port(s->ctx, 1, &s->port_attr)) {
    set_err("qes_server_open: ibv_query_port failed");
    qes_server_free(s);
    return nullptr;
  }
  s->pd = ibv_alloc_pd(s->ctx);
  if (!s->pd) {
    set_err("qes_server_open: ibv_alloc_pd failed");
    qes_server_free(s);
    return nullptr;
  }

  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  for (int i = 0; i < n_buckets; i++) {
    BucketEp& ep = s->eps[i];
    ep.rows = buckets[i];
    ep.port = ctl_port + 1 + i;
    /* This process's own caps: every session must match them (handle_ctl_conn), so the slabs are
     * laid out once, before any client is known. */
    const Geometry g = geometry(ep.rows, slots, wire_caps());
    ep.req_len = g.req_len;
    ep.ids_off = g.ids_off;
    ep.w_off = g.w_off;
    ep.resp_len = g.resp_len;
    ep.resp_off = g.resp_off;
    ep.stride = g.stride;
    ep.slab_bytes = g.slab_bytes;
    if (cudaHostAlloc(&ep.slab, ep.slab_bytes, cudaHostAllocMapped | cudaHostAllocPortable) !=
        cudaSuccess) {
      set_err("qes_server_open: cudaHostAlloc %zu bytes (rows=%d) failed", ep.slab_bytes, ep.rows);
      qes_server_free(s);
      return nullptr;
    }
    memset(ep.slab, 0, ep.slab_bytes);
    ep.mr = ibv_reg_mr(s->pd, ep.slab, ep.slab_bytes, access);
    if (!ep.mr) {
      set_err("qes_server_open: ibv_reg_mr %zu bytes (rows=%d) failed", ep.slab_bytes, ep.rows);
      qes_server_free(s);
      return nullptr;
    }
    /* Same geometry, server side: port, stride, registered length, base, rkey. */
    if (qes_trace())
      fprintf(stderr,
              "SEAM_TRACE_EP role=srv port=%d rows=%u slots=%d stride=%zu slab=%zu req=%zu "
              "resp=%zu base=0x%016llx rkey=0x%08x\n",
              ep.port, ep.rows, slots, ep.stride, ep.slab_bytes, ep.req_len, ep.resp_len,
              (unsigned long long)(uintptr_t)ep.slab, ep.mr->rkey);
  }

  /* Pinned slab memory, as laid out for this process's caps against the fp32 reply of v4: the
   * 16-bit reply halves every slot's reply region, per bucket and per slot (the P7 client lays its
   * slabs out from the same mask, so it saves the same bytes). */
  {
    size_t now = 0, f32 = 0;
    for (int i = 0; i < n_buckets; i++) {
      now += s->eps[i].slab_bytes;
      f32 += geometry(buckets[i], slots, 0).slab_bytes;
    }
    say("SEAM_SLAB_BYTES caps=0x%x out=%s slab_bytes=%zu f32_slab_bytes=%zu saved_bytes=%zu "
        "(per side; buckets=%s slots=%d)",
        (unsigned)wire_caps(), out_name(wire_caps()), now, f32, f32 - now,
        csv(buckets, n_buckets).c_str(), slots);
  }
  s->listen_fd = tcp_listen(s->ctl_port, &why);
  if (s->listen_fd < 0) {
    set_err("qes_server_open: control listen on %d failed: %s", s->ctl_port, why);
    qes_server_free(s);
    return nullptr;
  }

  /* Every bucket's data port is bound now, not when a session asks for it: the negotiation reply
   * hands the client all of them at once and it connects to them in request order, so a port that
   * is only bound later refuses that connection. */
  for (int i = 0; i < n_buckets; i++) {
    s->eps[i].ls = tcp_listen(s->eps[i].port, &why);
    if (s->eps[i].ls < 0) {
      set_err("qes_server_open: data listen on %d (rows=%d) failed: %s", s->eps[i].port,
              s->eps[i].rows, why);
      qes_server_free(s);
      return nullptr;
    }
  }

  say("SEAM_LISTEN ctl=%d buckets=%s ports=%d..%d slots=%d build=%s placement_sha=%s",
      s->ctl_port, csv(s->buckets, s->n_buckets).c_str(), s->ctl_port + 1,
      s->ctl_port + s->n_buckets, s->slots, s->build, s->placement_sha);
  return s;
}

extern "C" int qes_caps(void) { return wire_caps(); }

extern "C" int qes_out_mode(void) {
  const int c = wire_caps();
  if (c < 0) return -1;
  return (c & QES_CAP_OUT_BF16) ? 1 : (c & QES_CAP_OUT_FP16) ? 2 : 0;
}

extern "C" void qes_server_set_fn(qes_server* s, qes_server_fn fn, void* ctx) {
  if (!s) return;
  s->fn = fn;
  s->fn_ctx = ctx;
}

extern "C" int qes_server_run(qes_server* s) {
  if (!s) return -1;
  s->stopping.store(false);
  say("SEAM_SERVER_READY proto=%d ctl=%d buckets=%s slots=%d caps=0x%x out=%s", QES_PROTO_VERSION,
      s->ctl_port, csv(s->buckets, s->n_buckets).c_str(), s->slots, (unsigned)wire_caps(),
      out_name(wire_caps()));
  while (!s->stopping.load()) {
    pollfd fds[2];
    fds[0] = {s->listen_fd, POLLIN, 0};
    fds[1] = {-1, 0, 0};
    if (s->active && s->active->fd >= 0) fds[1] = {s->active->fd, POLLIN, 0};
    int r = poll(fds, 2, 250);
    if (r < 0 && errno != EINTR) {
      set_err("qes_server_run: poll failed: %s", strerror(errno));
      return -1;
    }
    if (r > 0) {
      if (fds[0].revents & POLLIN) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        int fd = accept(s->listen_fd, (sockaddr*)&peer, &plen);
        if (fd >= 0) {
          char ip[64] = "?";
          char buf[INET_ADDRSTRLEN] = {0};
          if (inet_ntop(AF_INET, &peer.sin_addr, buf, sizeof(buf)))
            snprintf(ip, sizeof(ip), "%s", buf);
          handle_ctl_conn(s, fd, ip);
        } else if (errno != EINTR) {
          set_err("qes_server_run: accept on control port failed: %s", strerror(errno));
        }
      } else if (fds[0].revents & (POLLERR | POLLNVAL)) {
        set_err("qes_server_run: control listener is gone");
        return -1;
      }
      if (fds[1].fd >= 0 && s->active && fds[1].fd == s->active->fd && fds[1].revents) {
        char b[1];
        ssize_t k = recv(s->active->fd, b, 1, MSG_DONTWAIT);
        if (k == 0)
          end_active(s, "client-closed");
        else if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
          end_active(s, "client-reset");
        else if (k > 0 && b[0] == 'B')
          end_active(s, "client-bye");
      }
    }
    reap_waiters(s);
    if (s->active && s->active->th_done.load()) end_active(s, "internal");
  }
  end_active(s, "server-stop");
  return 0;
}

extern "C" void qes_server_shutdown(qes_server* s) {
  if (s) s->stopping.store(true);
}

extern "C" void qes_server_free(qes_server* s) {
  if (!s) return;
  s->stopping.store(true);
  end_active(s, "server-free");
  for (auto& p : s->waiting)
    if (p.fd >= 0) close(p.fd);
  s->waiting.clear();
  if (s->listen_fd >= 0) {
    close(s->listen_fd);
    s->listen_fd = -1;
  }
  for (int i = 0; i < s->n_buckets; i++) {
    if (s->eps[i].ls >= 0) {
      close(s->eps[i].ls);
      s->eps[i].ls = -1;
    }
    if (s->eps[i].mr) ibv_dereg_mr(s->eps[i].mr);
    if (s->eps[i].slab) cudaFreeHost(s->eps[i].slab);
    s->eps[i].mr = nullptr;
    s->eps[i].slab = nullptr;
  }
  if (s->pd) ibv_dealloc_pd(s->pd);
  if (s->ctx) ibv_close_device(s->ctx);
  delete s;
}

extern "C" int qes_status(const char* host, int ctl_port, qes_nego* out) {
  if (out) memset(out, 0, sizeof(*out));
  if (!host || ctl_port <= 0) {
    set_err("qes_status: bad host/port");
    return -1;
  }
  const char* why = "";
  int fd = tcp_connect_wait(host, ctl_port, 5000, &why);
  if (fd < 0) {
    set_err("qes_status: cannot reach %s:%d: %s", host, ctl_port, why);
    return -1;
  }
  sock_opts(fd, 5, 5);
  int buckets[1] = {1};
  CtlHello hello;
  ctl_hello_init(&hello, "status", buckets, 1, QES_NL, CTL_FLAG_STATUS);
  CtlReply reply{};
  bool ok = xfer(fd, &hello, sizeof(hello), true, &why);
  if (ok) ok = xfer(fd, &reply, sizeof(reply), false, &why);
  close(fd);
  if (!ok) {
    set_err("qes_status: no reply from %s:%d: %s", host, ctl_port, why);
    return -1;
  }
  if (reply.magic != CTL_REPLY_MAGIC || reply.status != QES_ST_STATUS) {
    set_err("qes_status: %s:%d did not answer as a seam control port (magic=0x%08x status=%d)",
            host, ctl_port, reply.magic, reply.status);
    return -1;
  }
  if (out) nego_from_reply(reply, out);
  return 0;
}

/* =========================================================================================
 * Data path (per-call protocol unchanged)
 * ========================================================================================= */

/* A handle whose QP raised an error (or whose response never came) can never recover: say so in a
 * form the Python wrapper can raise, instead of letting the caller wait out a timeout or consume a
 * stale response and turn one dead bucket into a sticky CUDA error for the whole process. */
int seam_dead_check(qes_handle* h, const char* what) {
  if (!h->dead.load(std::memory_order_relaxed)) return 0;
  set_err("qes_%s: the seam handle rows=%d port=%d is dead (the transport failed, see the "
          "\"qes:\" line above this one)",
          what, h->rows, h->tcp_port);
  return -5;
}

/* rows_used: how many of the bucket's rows this request carries, or <= 0 for the whole bucket. The
 * bucket sizes the slab, the RDMA regions and every offset; the copies here, the compute on the
 * peer and the response follow rows_used (see SlotHdr::rows). A prefill chunk that the grant puts
 * in a 4096-row bucket while it only has 2560 rows therefore costs 2560 rows of wire and compute. */
extern "C" int qes_issue(qes_handle* h, int layer, const void* x_bf16, const int32_t* ids,
                         const void* w_bf16, cudaStream_t stream, int n_cold, int rows_used) {
  if (!h || h->role != QES_ROLE_CLIENT || layer < 0 || layer >= QES_NL) return -1;
  if (int rc = seam_dead_check(h, "issue")) return rc;
  if (h->n_pending >= h->slots) {
    set_err("qes_issue: %d request(s) already in flight on rows=%d (slots=%d): collect one first",
            h->n_pending, h->rows, h->slots);
    return -4;
  }
  /* Pick a free slot whose previous request the server has already answered. `pending` alone is not
   * enough: the entry leaves `pending` when the collect is only *enqueued*, so the slot's staging
   * bytes may still be the ones the NIC is reading while the next request's device-to-host copies
   * land on top of them (a 24-byte id copy interleaving with the payload the proxy is posting put
   * another layer's cold ids on the wire, measured 2026-11-01). The server reads the payload before
   * it posts the response, so `fresp >= issued` proves the old bytes are free.
   * `issued`, not the slab's F_REQ: F_REQ is bumped by a kernel on the caller's stream, so it still
   * reads 0 for a request the GPU has not reached yet, and the slot was then reused under a request
   * that had never been forwarded at all. */
  /* Under graph capture this host code runs once per layer and its result is baked: on every replay
   * the baked slot, the baked copies and the baked publish run again, but nothing here does. The
   * ack test therefore cannot be part of the decision -- nothing has executed yet, so F_RESP still
   * reads 0 and after `slots` layers every slot would look busy (a capture with `slots`=4 stopped at
   * layer 4 and would have waited out SLOT_ACK_TIMEOUT_NS). What stays valid is the rest of the rule:
   * `pending` is host state, and at replay the captured requests are stream-ordered, so a slot that
   * the capture had already closed cannot be reused by an earlier replayed layer. */
  int cap_status = 0;
  const bool capturing =
      stream &&
      cudaStreamIsCapturing(stream, (cudaStreamCaptureStatus*)&cap_status) == cudaSuccess &&
      cap_status == (int)cudaStreamCaptureStatusActive;
  int slot = -1;
  const int64_t t0 = now_ns();
  for (;;) {
    for (int k = 0; k < h->slots && slot < 0; k++) {
      const int cand = (h->next_slot + k) % h->slots;
      bool used = false;
      for (int p = 0; p < h->n_pending; p++)
        if (h->pending[p].slot == cand) used = true;
      if (used) continue;
      const uint64_t mine = h->issued[cand];
      if (!capturing && mine && load_acq(h->slab, off_fresp(h, cand)) < mine) continue;
      slot = cand;
    }
    if (slot >= 0) break;
    if (h->dead.load(std::memory_order_relaxed)) return seam_dead_check(h, "issue");
    if (now_ns() - t0 > SLOT_ACK_TIMEOUT_NS) {
      set_err("qes_issue: rows=%d port=%d: every slot is still unacknowledged after %llu s "
              "(in_flight=%d, slots=%d): the seam transport is stuck",
              h->rows, h->tcp_port, (unsigned long long)(SLOT_ACK_TIMEOUT_NS / 1000000000ull),
              h->n_pending, h->slots);
      return -4;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  SlotHdr hdr{};
  hdr.magic = SLOT_MAGIC;
  hdr.layer = (uint32_t)layer;
  const int nrows = rows_used > 0 && rows_used <= h->rows ? rows_used : h->rows;
  hdr.rows = (uint32_t)nrows;
  hdr.n_cold = n_cold > 0 ? (uint32_t)n_cold : 0u;
  /* The seq publish_kernel is about to produce for this slot (it does `*F_REQ + 1`). Both proxies
   * compare it against the seq they are handling, so a header that does not belong to the payload
   * is caught here instead of being computed with the wrong layer's ids. The value here only has to
   * be right for the eager path; publish_kernel rewrites the field on the device (see SlotHdr::seq),
   * which is what a replayed graph needs. */
  hdr.seq = load_acq(h->slab, off_freq(h, slot)) + 1;
  /* Written on the device, not staged in host memory: the destination is the wire, and a copy that
   * fills a host stage line runs at capture time only, so its content at replay is whatever the
   * *last capture* of that line left there. One line per layer is shared by every graph of that
   * layer, i.e. by every row count decode captures, so a replay of a narrower graph advertised the
   * wider graph's `rows` (measured 2026-11-02, gp-ab12: `rows=5 ... 2 of 30 pick(s) outside
   * [-1,251), first at 18` -- 18 is row 3 of a request that carried 3 rows, so the proxy
   * range-checked two rows no request had written and the GX would have computed them). Baked
   * kernel arguments are per graph, so a replay always writes its own constants; the write is
   * stream-ordered ahead of the publish below, which is the doorbell the proxy waits for (this is
   * also what the old per-layer line was for: a header from the layer being replayed, not from
   * whatever layer last wrote the slot -- 2026-11-01, "layer=19 slot=3 ... 268"). */
  hdr_kernel<<<1, 1, 0, stream>>>(h->slab + off_hdr(h, slot), hdr.magic, hdr.layer, hdr.rows,
                                 hdr.n_cold);
  if (cudaGetLastError() != cudaSuccess) return -3;
  uint8_t* req = h->slab + off_req(h, slot);
  if (cudaMemcpyAsync(req, x_bf16, (size_t)nrows * QES_HIDDEN * 2, cudaMemcpyDeviceToHost, stream))
    return -2;
  if (cudaMemcpyAsync(req + h->ids_off, ids, (size_t)nrows * QES_TOPK * 4,
                      cudaMemcpyDeviceToHost, stream))
    return -2;
  /* Same stream, same source pointer, and before the publish below: if the slab copy above is what
   * the wire carries, this mirror says whether the caller's tensor already held that value. */
  if (h->dump_src &&
      cudaMemcpyAsync(h->dump_src + (size_t)slot * h->rows * QES_TOPK, ids,
                      (size_t)nrows * QES_TOPK * 4, cudaMemcpyDeviceToHost, stream))
    return -2;
  if (cudaMemcpyAsync(req + h->w_off, w_bf16, (size_t)nrows * QES_TOPK * 2,
                      cudaMemcpyDeviceToHost, stream))
    return -2;
  if (qes_trace())
    fprintf(stderr,
            "SEAM_TRACE_REQ role=cli port=%d rows=%u layer=%d slot=%d off_hdr=%zu off_req=%zu "
            "req_len=%zu slab=%zu\n",
            h->tcp_port, h->rows, layer, slot, off_hdr(h, slot), off_req(h, slot), h->req_len,
            h->slab_bytes);
  publish_kernel<<<1, 1, 0, stream>>>(
      (uint64_t*)(h->slab + off_freq(h, slot)),
      (uint64_t*)(h->slab + off_hdr(h, slot) + offsetof(SlotHdr, seq)));
  if (cudaGetLastError() != cudaSuccess) return -3;
  /* Only after the publish is queued: this is what a later issue reads to decide the slot is free.
   * A captured issue leaves it alone: its baked seq is not the one the replay will publish, and the
   * slot's real state is read from F_RESP once the graph runs. */
  if (!capturing) h->issued[slot] = hdr.seq;
  h->pending[h->n_pending].layer = layer;
  h->pending[h->n_pending].slot = slot;
  h->n_pending++;
  h->next_slot = (slot + 1) % h->slots;
  return 0;
}

/* rows: the row count of the request being collected (the same value qes_issue was given), or <= 0
 * for the whole bucket. Only those rows are copied out: the peer answered exactly that many, so the
 * rest of the slot's response region holds an older request's rows and must not reach the caller. */
extern "C" int qes_collect(qes_handle* h, int layer, float* out_f32, cudaStream_t stream, int rows) {
  if (!h || h->role != QES_ROLE_CLIENT || layer < 0 || layer >= QES_NL) return -1;
  if (int rc = seam_dead_check(h, "collect")) return rc;
  int idx = -1;
  for (int p = 0; p < h->n_pending && idx < 0; p++)
    if (h->pending[p].layer == layer) idx = p;
  if (idx < 0) {
    set_err("qes_collect: no request in flight for layer %d on rows=%d (in_flight=%d)", layer,
            h->rows, h->n_pending);
    return -1;
  }
  const int slot = h->pending[idx].slot;
  const int nrows = rows > 0 && rows <= h->rows ? rows : h->rows;
  /* The reply width follows the session's caps. fp32 wire: one H2D copy of the slot, as in v4.
   * 16-bit wire: no copy and no device staging buffer - widen_kernel reads the bf16/fp16 rows
   * through the mapped slab and writes fp32 into the caller's out_f32, which therefore stays a
   * [rows,HIDDEN] fp32 buffer on both wires. It runs after wait_kernel on the same stream, so a
   * poisoned slot (NaN words in the reply's format) widens to fp32 NaN. */
  const size_t n_elem = (size_t)nrows * QES_HIDDEN;
  const size_t resp_bytes = n_elem * out_elem_of(h->caps);
  // A whole warp is enough to poison a timed-out response quickly (resp_len/4 up to ~10M words for
  // the largest bucket) and costs nothing on the answered path, where the loop is never entered.
  wait_kernel<<<1, 32, 0, stream>>>((const uint64_t*)(h->slab + off_freq(h, slot)),
                                    (const uint64_t*)(h->slab + off_fresp(h, slot)),
                                    (uint32_t*)(h->slab + off_fail(h, slot)),
                                    (uint32_t*)(h->slab + off_resp(h, slot)),
                                    (int)(resp_bytes / sizeof(uint32_t)), nan_word(h->caps),
                                    WAIT_TIMEOUT_NS);
  if (out_elem_of(h->caps) == 2) {
    const int threads = 256;
    const size_t n2 = (n_elem + 1) / 2;
    const unsigned blocks = (unsigned)std::min<size_t>((n2 + threads - 1) / threads, 65535u * 8u);
    widen_kernel<<<blocks, threads, 0, stream>>>(
        (const uint32_t*)(h->slab + off_resp(h, slot)), out_f32, n_elem,
        (h->caps & QES_CAP_OUT_BF16) != 0);
  } else if (cudaMemcpyAsync(out_f32, h->slab + off_resp(h, slot), resp_bytes,
                             cudaMemcpyHostToDevice, stream)) {
    return -2;
  }
  for (int p = idx; p + 1 < h->n_pending; p++) h->pending[p] = h->pending[p + 1];
  h->n_pending--;
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

/* Debug: dump this handle's slab to a file. Offsets only mean something next to Geometry above;
 * used to prove whether a response the peer says it posted is actually in the slab the client
 * copies from (a zeroed response region and a healthy flag word cannot be told apart otherwise). */
extern "C" int qes_debug_dump(qes_handle* h, const char* path) {
  if (!h || !path) return -1;
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  const size_t n = fwrite(h->slab, 1, h->slab_bytes, f);
  fclose(f);
  return n == h->slab_bytes ? 0 : -2;
}

extern "C" void qes_destroy(qes_handle* h) { stop_handle(h); }
