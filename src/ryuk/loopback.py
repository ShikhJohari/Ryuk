"""What counts as this machine."""

from ipaddress import ip_address


def is_loopback_host(host: str) -> bool:
    """True for `localhost` or a loopback IP address (no port).

    Hostnames are case-insensitive and may end in a root dot (`localhost.`).
    """
    if host.lower().removesuffix(".") == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
