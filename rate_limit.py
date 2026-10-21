"""
rate_limit.py - Protects the Tavily credit quota from spam.

Every request that reaches /ask or /search costs one Tavily credit, so each
request is checked against three fixed-window limits BEFORE any search runs:

    ip_minute   - requests per IP per minute (stops a single fast script)
    ip_day      - requests per IP per 24 hours (stops one IP from eating the
                  global daily cap and locking everyone else out)
    global_day  - requests per UTC day across all IPs (hard ceiling on credits)

The counters live in Redis (Upstash), so they survive Render restarts. One Lua
script checks all three and increments them only if every check passes, so a
rejected request never consumes quota and there is no race between instances.

A limit set to 0 is disabled. If REDIS_URL is empty, rate limiting is off.
"""

import logging
from datetime import datetime, timedelta, timezone

import redis
from fastapi import HTTPException, Request

from config import (
    RATE_LIMIT_DEBUG,
    RATE_LIMIT_FAIL_OPEN,
    RATE_LIMIT_GLOBAL_PER_DAY,
    RATE_LIMIT_IP_PER_DAY,
    RATE_LIMIT_IP_PER_MINUTE,
    REDIS_URL,
    TRUSTED_PROXY_HOPS,
)

logger = logging.getLogger("rate_limit")

MINUTE_SECONDS = 60
DAY_SECONDS = 24 * 60 * 60

# KEYS: [1] ip_minute, [2] ip_day, [3] global_day
# ARGV: limit_ip_minute, limit_ip_day, limit_global_day, ttl_minute, ttl_day
# Returns {1, "", 0} when allowed, or {0, <name of the limit hit>, <seconds left>}.
_LUA_CHECK_AND_INCREMENT = """
local limits = {tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])}
local ttls = {tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[5])}
local names = {'ip_minute', 'ip_day', 'global_day'}

for i = 1, 3 do
  if limits[i] > 0 then
    local current = tonumber(redis.call('GET', KEYS[i]) or '0')
    if current >= limits[i] then
      local ttl = redis.call('TTL', KEYS[i])
      if ttl < 1 then ttl = 1 end
      return {0, names[i], ttl}
    end
  end
end

for i = 1, 3 do
  if limits[i] > 0 then
    local count = redis.call('INCR', KEYS[i])
    if count == 1 then redis.call('EXPIRE', KEYS[i], ttls[i]) end
  end
end
return {1, '', 0}
"""


class RateLimitUnavailable(Exception):
    """Redis could not be reached, so the limits could not be checked."""


class RateLimiter:
    def __init__(
        self,
        client,
        ip_per_minute: int,
        ip_per_day: int,
        global_per_day: int,
    ):
        self._client = client
        self._limits = (ip_per_minute, ip_per_day, global_per_day)
        self._script = client.register_script(_LUA_CHECK_AND_INCREMENT)

    def check(self, ip: str, now: datetime | None = None) -> tuple[bool, str, int]:
        """Return (allowed, limit_hit, retry_after_seconds)."""
        now = now or datetime.now(timezone.utc)
        keys = [
            f"rl:ip_minute:{ip}",
            f"rl:ip_day:{ip}",
            f"rl:global_day:{now:%Y%m%d}",
        ]
        args = [*self._limits, MINUTE_SECONDS, DAY_SECONDS]
        try:
            allowed, limit_hit, ttl = self._script(keys=keys, args=args)
        except redis.RedisError as e:
            raise RateLimitUnavailable(f"{type(e).__name__}: {e}") from e

        if allowed == 1:
            return True, "", 0

        if isinstance(limit_hit, bytes):
            limit_hit = limit_hit.decode()
        if limit_hit == "global_day":
            # The global counter is keyed by UTC date: it resets at midnight UTC.
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            ttl = max(1, int((tomorrow - now).total_seconds()))
        return False, limit_hit, int(ttl)


def get_client_ip(request: Request, trusted_hops: int) -> str:
    """Best-effort real client IP behind reverse proxies.

    Each trusted proxy appends the address of the peer it received the request
    from. The LEFT part of X-Forwarded-For can be forged by the client, so the
    address is read from the RIGHT: with N trusted proxies the client is the
    N-th entry from the end. trusted_hops=0 ignores the header entirely.
    """
    direct = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return direct
    header = request.headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in header.split(",") if p.strip()]
    if len(parts) >= trusted_hops:
        return parts[-trusted_hops]
    return direct


def _build_limiter() -> RateLimiter | None:
    if not REDIS_URL:
        logger.warning("REDIS_URL is not set: rate limiting is OFF")
        return None
    client = redis.Redis.from_url(
        REDIS_URL, socket_timeout=2, socket_connect_timeout=2
    )
    return RateLimiter(
        client,
        ip_per_minute=RATE_LIMIT_IP_PER_MINUTE,
        ip_per_day=RATE_LIMIT_IP_PER_DAY,
        global_per_day=RATE_LIMIT_GLOBAL_PER_DAY,
    )


# Module-level so tests can swap it for a limiter backed by a fake Redis.
limiter: RateLimiter | None = _build_limiter()

_MESSAGES = {
    "ip_minute": "Too many requests from your connection. Please wait a moment and try again.",
    "ip_day": "You have reached the daily limit for this demo. Please try again tomorrow.",
    "global_day": "This demo has reached its daily search budget. Please try again tomorrow.",
}


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: raises 429 when a limit is hit, 503 if the limiter
    is down and RATE_LIMIT_FAIL_OPEN is false."""
    if limiter is None:
        return

    ip = get_client_ip(request, TRUSTED_PROXY_HOPS)
    if RATE_LIMIT_DEBUG:
        logger.info(
            "rate-limit ip=%s x-forwarded-for=%r client.host=%r",
            ip,
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
        )

    try:
        allowed, limit_hit, retry_after = limiter.check(ip)
    except RateLimitUnavailable as e:
        if RATE_LIMIT_FAIL_OPEN:
            logger.warning("Rate limiter unavailable, allowing request: %s", e)
            return
        logger.error("Rate limiter unavailable, rejecting request: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"code": "rate_limiter_unavailable", "message": "The service is temporarily unavailable."},
        )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limited",
                "limit": limit_hit,
                "retry_after": retry_after,
                "message": _MESSAGES[limit_hit],
            },
            headers={"Retry-After": str(retry_after)},
        )
