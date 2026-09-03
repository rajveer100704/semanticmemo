import socket
import sys

import psycopg
import redis
from qdrant_client import QdrantClient


def check_all():
    print("=== LIVE SERVICE HEALTH CHECK ===")

    # 1. PostgreSQL
    try:
        dsn = "postgresql://postgres:postgres_dev_password@localhost:5432/semanticmemo"
        with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT version();")
            v = cur.fetchone()[0]
            print(f"[PASS] PostgreSQL: {v.split(',')[0]}")
    except Exception as e:
        print(f"[FAIL] PostgreSQL: {e}")
        return False

    # 2. Redis
    try:
        r = redis.Redis.from_url("redis://localhost:6379/0", socket_timeout=3)
        res = r.ping()
        info = r.info("server")
        print(f"[PASS] Redis: ping={res}, version={info.get('redis_version')}")
        r.close()
    except Exception as e:
        print(f"[FAIL] Redis: {e}")
        return False

    # 3. Qdrant
    try:
        qc = QdrantClient(url="http://localhost:6333", timeout=3, check_compatibility=False)
        cols = qc.get_collections()
        print(f"[PASS] Qdrant: reachable, collections={len(cols.collections)}")
        qc.close()
    except Exception as e:
        print(f"[FAIL] Qdrant: {e}")
        return False

    # 4. Kafka
    try:
        s = socket.create_connection(("localhost", 9092), timeout=3)
        s.close()
        print("[PASS] Kafka: port 9092 reachable")
    except Exception as e:
        print(f"[FAIL] Kafka: {e}")
        return False

    print("ALL 4 LIVE SERVICES HEALTHY AND REACHABLE!")
    return True


if __name__ == "__main__":
    ok = check_all()
    sys.exit(0 if ok else 1)
