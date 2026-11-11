/* qmesh_expert_seam: P7 <-> GX1 expert-offload seam over host-staged RC RDMA.
 *
 * Session model (protocol v3)
 * --------------------------
 * The GX server loads the cold experts once and then serves client sessions forever. One TCP
 * control port (default 47311) carries the versioned hello/negotiation; the per-bucket data
 * handles (one RC QP per bucket, as before) live on ports assigned by the server and returned in
 * the negotiation reply. A session ends when the client closes the control socket (clean exit,
 * ctrl-c or kill -9 all close it), at which point the server destroys that session's QPs and
 * serves the next client with no reload. Handshake/protocol errors reject one connection with a
 * logged reason and never abort the process.
 *
 * Client (P7):
 *   qes_client_session* cs = qes_client_open(peer, 47311, build, buckets, n, QES_NL, wait, &nego);
 *   qes_handle* h = qes_client_handle(cs, dev_name, gid_idx, rows, device_ordinal);
 *       qes_issue(h, L, x, ids, w, stream, n_cold, rows) / qes_collect(h, L, out, stream, rows)
 *   qes_client_close(cs);
 *
 * Server (GX1):
 *   qes_server* s = qes_server_open(dev, gid, 47311, 0, buckets, n, build, placement_sha,
 *                                   allow_wait, slots, &err);
 *   qes_server_set_fn(s, fn, ctx);          // optional compute hook, called from proxy threads
 *   qes_server_run(s);                      // serves sessions until qes_server_shutdown(s)
 *
 * Per-call wire: for each request an RC RDMA WRITE of [64 B header | x | ids | w] followed by an
 * 8-byte RDMA WRITE of that slot's sequence flag on the same QP; the reply is an RDMA WRITE of
 * out followed by the slot's reply flag. RC delivers in order, so a flag observed by the peer
 * implies its payload is visible. The header carries the layer index and row count, so one
 * payload slot serves any layer (protocol v3; before v3 a slot existed per layer, which cost 40x
 * the staging memory and capped the usable row bucket).
 *
 * Client (P7) role 0 issues per-layer requests from CUDA streams/graphs:
 *   x:   bf16 [rows,5120] device ptr, ids: int32 [rows,6] device ptr (cold-local ids, -1 = hot),
 *   w:   bf16 [rows,6] device ptr, out: fp32 [rows,5120] device ptr on every wire (under the
 *        16-bit caps collect widens the bf16/fp16 slot into it with one kernel, exactly).
 *
 * Wire caps (protocol v5): each half reads SEAM_QES_OUT=f32|bf16|fp16 from its own environment and
 * the resulting mask rides the control hello, the negotiation reply and the data hello. The mask
 * decides the reply slot's element size, so a session whose masks differ is refused with both named;
 * nothing is negotiated down. Unset is the v4 fp32 reply, byte for byte. Any other word refuses to
 * open (server and client), so a typo cannot measure the previous wire.
 *
 * issue/collect are kernel launches; sequence counters live in mapped host memory, so the calls are
 * capturable in CUDA graphs. A handle owns `slots` payload slots (default 1) and issues into the
 * next free one; with one slot issue/collect pairs strictly serialize, which is what the client
 * policy does. A handle with more slots may have that many requests in flight.
 *
 * Memory: pinned host memory (cudaHostAllocMapped). No GPUDirect RDMA: it is blocked on both hosts
 * (nvidia-peermem absent on P7, ibv_reg_mr EFAULT on GX1).
 */
#ifndef QMESH_EXPERT_SEAM_H
#define QMESH_EXPERT_SEAM_H

#include <cuda_runtime.h>
#include <stdint.h>

#define QES_NL 40
#define QES_HIDDEN 5120
#define QES_TOPK 6

/* Control protocol version. Both sides must agree; a mismatch is reported on the client with
 * both versions and build ids as the message text. v4: the request header carries the seq it was
 * written for (SlotHdr.seq) and both proxies require it to match the flag they are handling.
 * v5: the wire caps mask (QES_CAP_*) rides the control hello, the negotiation reply and the data
 * hello, and it decides the reply slot geometry on both sides. */
#define QES_PROTO_VERSION 5
#define QES_MAX_BUCKETS 12
#define QES_MAX_SLOTS 8     /* payload slots per data handle (in-flight requests per bucket) */
#define QES_BUILD_LEN 40
#define QES_MSG_LEN 192

/* Wire caps (protocol v5): what the reply bytes mean. At most one OUT bit is set. */
#define QES_CAP_OUT_BF16 1 /* reply: bf16 [rows,HIDDEN] instead of fp32 */
#define QES_CAP_OUT_FP16 2 /* reply: fp16 [rows,HIDDEN] instead of fp32 */

#if defined(_WIN32)
#define QES_API __declspec(dllexport)
#else
#define QES_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qes_handle qes_handle;
typedef struct qes_client_session qes_client_session;
typedef struct qes_server qes_server;

#define QES_ROLE_CLIENT 0
#define QES_ROLE_SERVER 1

/* Negotiation outcome (client) / error detail (server). */
#define QES_ST_OK 0         /* session granted */
#define QES_ST_BUSY 1       /* another session is active and wait was not requested */
#define QES_ST_QUEUED 2     /* wait requested, session queued behind the active one */
#define QES_ST_INCOMPAT 3   /* protocol version or layer count mismatch */
#define QES_ST_BADREQUEST 4 /* no common row bucket, or a malformed hello */
#define QES_ST_ERROR 5      /* internal server-side failure while admitting the session */
#define QES_ST_STATUS 6     /* reply to a status query: no session was started */

typedef struct {
  int32_t status;      /* QES_ST_* */
  int32_t slots;       /* payload slots per data handle: the client sizes its slabs with this */
  int32_t version;     /* server protocol version */
  int32_t layers;      /* server QES_NL */
  int32_t n_buckets;   /* buckets the client may use in this session */
  int32_t n_requested; /* buckets the client asked for */
  int32_t buckets[QES_MAX_BUCKETS];
  int32_t ports[QES_MAX_BUCKETS]; /* data port for buckets[i] */
  int32_t requested[QES_MAX_BUCKETS];
  int32_t caps;                      /* wire caps of the session (QES_CAP_*, protocol v5) */
  char build[QES_BUILD_LEN];         /* server build/commit id */
  char placement_sha[QES_BUILD_LEN]; /* server placement fingerprint */
  char message[QES_MSG_LEN];         /* human-readable detail; set on every non-OK status */
} qes_nego;

/* Human-readable detail for the last failed call on this thread. */
QES_API const char* qes_last_error(void);

/* ---- client ---------------------------------------------------------------------------- */

/* Negotiates a session with a running server. `buckets` are the row buckets the client would like
 * to use; the server answers with the intersection it can serve (adopted by the caller).
 * wait != 0 asks the server to queue this client behind the active session instead of failing.
 * Returns NULL on any failure with the reason in `out->message` (and qes_last_error()). */
QES_API qes_client_session* qes_client_open(const char* peer_ip, int ctl_port, const char* build_id,
                                           const int* buckets, int n_buckets, int layers, int wait,
                                           qes_nego* out);

/* Data handle for one negotiated bucket on `peer_ip:nego.ports[i]`. NULL on failure, with the
 * reason in qes_last_error() (the server keeps serving). */
QES_API qes_handle* qes_client_handle(qes_client_session* s, const char* dev_name, int gid_idx,
                                      int bucket, int device_ordinal);

/* Ends the session (server frees that session's QPs and takes the next client). Destroys any
 * handles the caller did not destroy. */
QES_API void qes_client_close(qes_client_session* s);

/* ---- server ---------------------------------------------------------------------------- */

/* Opens the RDMA device and the control listener, and pins one staging slab per bucket (the
 * slabs and their MRs are reused by every session). buckets must be ascending and unique.
 * `slots` (1..QES_MAX_SLOTS) is the number of payload slots per handle: it is advertised to the
 * client, which sizes its own slabs from it, so both sides must agree (a mismatch is rejected at
 * the data handshake).
 * Returns NULL on a local failure (reason in qes_last_error(), details in `err`). */
QES_API qes_server* qes_server_open(const char* dev_name, int gid_idx, int ctl_port,
                                    int device_ordinal, const int* buckets, int n_buckets,
                                    const char* build_id, const char* placement_sha, int allow_wait,
                                    int slots, qes_nego* err);

/* Optional server-side compute hook, used by every session handle the server creates. Pointers
 * are into the pinned slab for layer L: x bf16 [rows,QES_HIDDEN], ids int32 [rows,QES_TOPK]
 * (cold-local, -1 = not on this host), w bf16 [rows,QES_TOPK], out [rows,QES_HIDDEN] in the reply
 * format of qes_out_mode() (f32 by default, bf16 or fp16 under the 16-bit caps; must be fully
 * written on return). Set before qes_server_run.
 *
 * The return value says whether the response region holds an answer, not just that the call ran:
 * 0 = answered, post it; nonzero = this request cannot be answered right now (the box is out of
 * memory, the compute raised) and the response must not be sent. The proxy then writes that code
 * into the client's slot fail word, releases the waiter's flag, ends this one session and keeps
 * listening. A request the GX cannot compute is not a reason to kill a server that holds a 113 GiB
 * pack and takes 150-220 s to reload. Codes, as the client's fail word receives them: 4 = out of
 * memory, 5 = compute failed while the CUDA context is still usable. */
typedef int (*qes_server_fn)(void* ctx, int layer, int rows, const void* x_bf16,
                             const int32_t* ids, const void* w_bf16, float* out_f32);
QES_API void qes_server_set_fn(qes_server* s, qes_server_fn fn, void* ctx);

/* This process's wire caps from SEAM_QES_OUT (QES_CAP_*), or -1 when the word is not one of
 * f32|bf16|fp16; qes_server_open and qes_client_open refuse to open with -1. Read once. */
QES_API int qes_caps(void);

/* The reply format the caps select, in the C hook's out_mode numbering: 0 = fp32, 1 = bf16,
 * 2 = fp16 (-1 when qes_caps() is -1). A hook writing `out` must use this format. */
QES_API int qes_out_mode(void);

/* Serves sessions until qes_server_shutdown. Returns 0 on clean shutdown, -1 on a fatal local
 * error (listener gone); session-level problems never return here. */
QES_API int qes_server_run(qes_server* s);

/* Asks qes_server_run to return: tears down the active session and closes the listener.
 * Safe to call from another thread. */
QES_API void qes_server_shutdown(qes_server* s);

/* Releases the RDMA context, slabs and listener. qes_server_run must have returned. */
QES_API void qes_server_free(qes_server* s);

/* ---- status ---------------------------------------------------------------------------- */

/* Queries the control port without starting (or disturbing) a session. Returns 0 and fills `out`
 * when the server answered, -1 with qes_last_error() otherwise. out->status is QES_ST_STATUS;
 * out->message describes the server state ("idle" or the busy session). */
QES_API int qes_status(const char* host, int ctl_port, qes_nego* out);

/* ---- data path ------------------------------------------------------------------------- */

/* n_cold: the layer's cold-expert count on the caller's placement, or 0 when unknown. It rides in the
 * request header and the client proxy refuses to send a pick outside [-1, n_cold), so a bad id takes
 * this client down instead of the server (client_check_ids in the .cu). */
/* rows: how many of the handle's bucket rows this request really carries, or 0 for the whole bucket.
 * The bucket sizes the slab and the wire geometry; it does not have to size the compute. The server
 * serves `rows` rows and answers `rows` rows, so a prefill chunk that lands in a bigger bucket no
 * longer costs the GX the padding: a 2560-row chunk in a 4096-row bucket computes 2560 rows. The
 * default of 0 is what the whole-bucket bench and every caller that predates this field mean. */
QES_API int qes_issue(qes_handle* h, int layer, const void* x_bf16, const int32_t* ids,
                      const void* w_bf16, cudaStream_t stream, int n_cold, int rows = 0);
QES_API int qes_collect(qes_handle* h, int layer, float* out_f32, cudaStream_t stream,
                        int rows = 0);

QES_API void qes_destroy(qes_handle* h);

#ifdef __cplusplus
}
#endif

#endif
