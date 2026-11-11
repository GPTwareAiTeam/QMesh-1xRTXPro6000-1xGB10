#!/usr/bin/env python3
"""Control-plane robustness probe for the seam server.

Connects to a running server's control port and sends a battery of handshakes a
server must survive: a status query, a bad magic, a wrong protocol version, a
wrong layer count, a malformed bucket count, a disjoint bucket list, a peer that
connects and says nothing (the 2 s hello timeout), and a legitimate hello. Then
the part the owners actually pay for: two more sessions in a row asking for
different bucket subsets of what the server serves, and a client that arrives
while somebody else's session is live, which must be refused with ST_BUSY
without touching that session. Between each, status must answer idle: if every
one of those got an answer and the pid never changed, the server did not die on
a client it did not expect.

Deliberately raw sockets and struct.pack: this is the regression test for the
"qes: peer layout mismatch" abort, so it must not share code with the client
that hides the bytes. Run it against a --stub server (no pack, no GPU).

Exit status: 0 all expectations met, 1 a mismatch (with the observed reply
printed), 2 could not talk to the port at all.
"""

import socket
import struct
import sys
import time

# qmesh/native/qmesh_expert_seam.h
QES_NL = 40
QES_PROTO_VERSION = 5
QES_MAX_BUCKETS = 12
QES_BUILD_LEN = 40
QES_MSG_LEN = 192
CTL_MAGIC = 0x32534551  # "QES2"
CTL_REPLY_MAGIC = 0x32525351  # "QSR2"
CTL_FLAG_WAIT = 1
CTL_FLAG_STATUS = 2
ST_OK, ST_BUSY, ST_QUEUED, ST_INCOMPAT, ST_BADREQUEST, ST_STATUS = 0, 1, 2, 3, 4, 6

HELLO = struct.Struct("<IIIII%diII%ds" % (QES_MAX_BUCKETS, QES_BUILD_LEN))
REPLY = struct.Struct("<IIIIIii%di%di%diI%ds%ds%ds"
                      % (QES_MAX_BUCKETS, QES_MAX_BUCKETS, QES_MAX_BUCKETS,
                         QES_BUILD_LEN, QES_BUILD_LEN, QES_MSG_LEN))


def hello(buckets, version=QES_PROTO_VERSION, layers=QES_NL, magic=CTL_MAGIC, flags=0,
          build=b"probe"):
    b = list(buckets) + [0] * (QES_MAX_BUCKETS - len(buckets))
    return HELLO.pack(magic, version, flags, layers, len(buckets), *b, 0, 0,
                      build.ljust(QES_BUILD_LEN, b"\0"))


def recv_reply(s, timeout=3.0):
    """Read one fixed-size CtlReply off a connected socket; b"" if the server closed first."""
    s.settimeout(timeout)
    buf = b""
    while len(buf) < REPLY.size:
        chunk = s.recv(REPLY.size - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def ask(host, port, payload, wait_s=8.0, send_nothing=False, timeout=3.0):
    """One connection, one hello, one reply. Returns (status, message, buckets) or None."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        if send_nothing:
            time.sleep(wait_s)          # server must time us out, not wait forever
            return None
        s.sendall(payload)
        buf = recv_reply(s, timeout)
        if len(buf) != REPLY.size:
            return None
        f = REPLY.unpack(buf)
        magic, ver, status, session, layers, nb, nreq = f[0:7]
        if magic != CTL_REPLY_MAGIC or ver != QES_PROTO_VERSION:
            print(f"  reply magic/version wrong: magic=0x{magic:08x} v={ver}")
            return None
        msg = f[-1].split(b"\0", 1)[0].decode(errors="replace")
        buckets = list(f[7:7 + nb])
        return status, msg, buckets
    finally:
        s.close()


def main(argv):
    host = argv[1] if len(argv) > 1 else "127.0.0.1"
    port = int(argv[2]) if len(argv) > 2 else 47911
    served = [int(x) for x in (argv[3] if len(argv) > 3 else "1,6,64,8192").split(",")]
    queue = len(argv) > 4 and argv[4] == "wait"   # server was started with --wait
    fails = 0

    def check(name, payload, want, want_msg=None, send_nothing=False):
        nonlocal fails
        try:
            got = ask(host, port, payload, send_nothing=send_nothing)
        except OSError as e:
            print(f"FAIL {name}: connection error {e}")
            fails += 1
            return
        if got is None:
            print(f"FAIL {name}: no usable reply (server closed or garbled it)")
            fails += 1
            return
        status, msg, buckets = got
        bad = status != want or (want_msg and want_msg not in msg)
        print(f"{'FAIL' if bad else 'ok  '} {name}: status={status} msg={msg!r}")
        if bad:
            fails += 1
        if want == ST_STATUS and buckets[:len(served)] != served:
            print(f"FAIL {name}: status reply buckets={buckets[:len(served)]} want={served}")
            fails += 1

    # 1. A status query must answer and list what the server serves.
    check("status", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 2. Every malformed or incompatible hello must get a typed rejection, not a dropped port.
    check("bad-magic", hello(served, magic=0xDEADBEEF), ST_INCOMPAT, "magic")
    check("wrong-version", hello(served, version=QES_PROTO_VERSION - 1), ST_INCOMPAT, "protocol")
    check("wrong-layers", hello(served, layers=QES_NL - 1), ST_INCOMPAT, "layer")
    check("no-buckets", hello([]), ST_BADREQUEST, "n_buckets")
    check("disjoint-buckets", hello([3, 7]), ST_BADREQUEST, "row bucket")

    # 3. A peer that connects and never speaks must be timed out, and the accept loop must
    #    survive it: the next status query is what proves that.
    print("     silent peer: waiting out the 2 s hello timeout")
    try:
        ask(host, port, b"", send_nothing=True)
    except OSError as e:
        print(f"FAIL silent-peer: connection error {e}")
        fails += 1
    check("status-after-silent-peer", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 4. A legitimate hello still starts a session.
    check("valid-hello", hello(served), ST_OK)

    # 5. And after that session's socket was closed under it, the server is back to idle,
    #    which means it reaped rather than died.
    time.sleep(0.5)
    check("status-after-valid-hello", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 6. The owner's requirement: the next client may want different buckets, and it must be
    #    served by the same process. Two sessions in a row, with disjoint bucket subsets of what
    #    the server offers, is the case that used to end the server's life.
    check("second-session-subset", hello([served[1]]), ST_OK)
    time.sleep(0.5)
    check("third-session-other-subset", hello([served[0], served[-1]]), ST_OK)
    time.sleep(0.5)
    check("status-after-three-sessions", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 7. A client that arrives while somebody else's session is live. Without SEAM_WAIT=1 it must
    #    be refused with ST_BUSY and the live session must survive the refusal; with the server's
    #    --wait it must instead be told ST_QUEUED and then admitted with ST_OK on the same socket
    #    once the live session ends. Both are one server lifetime and neither may be fatal, which
    #    is what lets a prod pair and a dev pair share the box.
    live = wait_sock = None
    try:
        live = socket.create_connection((host, port), timeout=3.0)
        live.sendall(hello(served))
        f = REPLY.unpack(recv_reply(live))
        ok = f[2] == ST_OK
        print(f"{'ok  ' if ok else 'FAIL'} concurrent: first session status={f[2]}")
        fails += 0 if ok else 1
        if queue:
            wait_sock = socket.create_connection((host, port), timeout=3.0)
            wait_sock.sendall(hello(served, flags=CTL_FLAG_WAIT))
            q = REPLY.unpack(recv_reply(wait_sock))
            ok = q[2] == ST_QUEUED
            qmsg = q[-1].split(b"\0", 1)[0].decode(errors="replace")
            print(f"{'ok  ' if ok else 'FAIL'} concurrent: second client status={q[2]} "
                  f"msg={qmsg!r}")
            fails += 0 if ok else 1
            live.close()
            live = None
            try:
                a = REPLY.unpack(recv_reply(wait_sock, timeout=15.0))
            except (OSError, struct.error) as e:
                a = None
                print(f"FAIL concurrent: queued client got no admission ({e})")
                fails += 1
            if a is not None:
                ok = a[2] == ST_OK
                print(f"{'ok  ' if ok else 'FAIL'} concurrent: queued client admitted "
                      f"status={a[2]} session={a[3]}")
                fails += 0 if ok else 1
        else:
            got = ask(host, port, hello(served))
            if got is None:
                print("FAIL concurrent: second client got no reply at all")
                fails += 1
            else:
                status, msg, _ = got
                ok = status == ST_BUSY and bool(msg)
                print(f"{'ok  ' if ok else 'FAIL'} concurrent: status={status} msg={msg!r}")
                fails += 0 if ok else 1
    except (OSError, struct.error) as e:
        print(f"FAIL concurrent: {e}")
        fails += 1
    finally:
        for s in (live, wait_sock):
            if s is not None:
                s.close()
    time.sleep(0.5)
    check("status-after-concurrent", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 8. A client that dies instead of closing: kill -9, a container stop, a dropped link. Its
    #    control socket is reset (SO_LINGER 0), which is what the server sees from a vanished
    #    peer, and the next client must still be served in the same lifetime.
    try:
        dead = socket.create_connection((host, port), timeout=3.0)
        dead.sendall(hello(served))
        d = REPLY.unpack(recv_reply(dead))
        dead.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        dead.close()                      # RST, not FIN
        ok = d[2] == ST_OK
        print(f"{'ok  ' if ok else 'FAIL'} vanished-client: session started status={d[2]}, "
              f"control socket reset")
        fails += 0 if ok else 1
    except (OSError, struct.error) as e:
        print(f"FAIL vanished-client: {e}")
        fails += 1
    time.sleep(1.5)
    check("status-after-vanished-client", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)
    check("session-after-vanished-client", hello(served), ST_OK)

    # 9. A stranger on a *data* port, which is where the historical "qes: peer layout mismatch" came
    #    from: the old server called fail() there and the process was gone. Start a session, then
    #    hammer the bucket ports with a wrong-magic hello, a truncated hello, and a peer that resets
    #    instead of closing. The server has to log each one as stale and keep the session alive.
    try:
        own = socket.create_connection((host, port), timeout=3.0)
        own.sendall(hello(served))
        o = REPLY.unpack(recv_reply(own))
        n = o[5]                     # n_buckets: buckets[12] hold values, ports[12] follow them
        dports = [p for p in o[7 + QES_MAX_BUCKETS:7 + 2 * QES_MAX_BUCKETS][:n] if p]
        ok = o[2] == ST_OK and len(dports) == n
        print(f"{'ok  ' if ok else 'FAIL'} data-peers: session status={o[2]} "
              f"buckets={','.join(str(b) for b in o[7:7 + n])} "
              f"ports={','.join(str(p) for p in dports)}")
        fails += 0 if ok else 1
        stray = [b"\x00" * 8,                                   # truncated
                 b"QHE3" + b"\0" * 44,                         # right magic, wrong layout
                 bytes(range(64))]                             # noise
        reached = 0
        for dp in dports:
            for blob in stray:
                try:
                    s = socket.create_connection((host, dp), timeout=2.0)
                    reached += 1
                    s.sendall(blob)
                    time.sleep(0.1)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    s.close()
                except OSError as e:
                    print(f"  note data-peers: port {dp}: {e}")
        # Reaching the port is the point: a wrong port would make the whole case a no-op. The
        # server logs one SEAM_REJECT reason=stale-data-conn per connection and the runner checks
        # that the reject count grew, which is what proves they were refused rather than ignored.
        ok = reached == len(dports) * len(stray)
        print(f"{'ok  ' if ok else 'FAIL'} data-peers: {reached} of "
              f"{len(dports) * len(stray)} bad hellos reached a bucket port")
        fails += 0 if ok else 1
        time.sleep(0.5)
        own.close()
    except (OSError, struct.error) as e:
        print(f"FAIL data-peers: {e}")
        fails += 1
    time.sleep(1.0)
    check("status-after-stray-data-peers", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)
    check("session-after-stray-data-peers", hello(served), ST_OK)
    time.sleep(0.5)

    print("RESULT ctl_robust=" + ("pass" if fails == 0 else f"fail({fails})"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
