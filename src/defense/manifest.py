"""Tool Capability Manifest and Security Constraints.

Defines tool categories, authorization policies, and argument constraints:
1. Tool Classification: READ_ONLY, NETWORK_READ, MUTATING, UNKNOWN.
2. Default-Deny: Any unregistered mutating tool is blocked by default.
3. Network Read Constraints: Strict host, scheme, and SSRF validation for fetch_page.
"""

from __future__ import annotations

import enum
import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


class ToolCategory(str, enum.Enum):
    READ_ONLY = "read_only"
    NETWORK_READ = "network_read"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    description: str
    required_args: list[str] = field(default_factory=list)
    sensitive: bool = False


# Canonical tool specifications for the agent environment
TOOL_MANIFEST: dict[str, ToolSpec] = {
    # Read-Only Tools (safe, local)
    "read_email": ToolSpec(
        name="read_email",
        category=ToolCategory.READ_ONLY,
        description="Read an email by ID",
        required_args=["email_id"],
        sensitive=False,
    ),
    "list_emails": ToolSpec(
        name="list_emails",
        category=ToolCategory.READ_ONLY,
        description="List emails in inbox",
        required_args=[],
        sensitive=False,
    ),
    "get_email": ToolSpec(
        name="get_email",
        category=ToolCategory.READ_ONLY,
        description="Get email details",
        required_args=["email_id"],
        sensitive=False,
    ),
    "read_file": ToolSpec(
        name="read_file",
        category=ToolCategory.READ_ONLY,
        description="Read file contents by path",
        required_args=["path"],
        sensitive=False,
    ),
    "list_files": ToolSpec(
        name="list_files",
        category=ToolCategory.READ_ONLY,
        description="List files in directory",
        required_args=[],
        sensitive=False,
    ),
    "get_event": ToolSpec(
        name="get_event",
        category=ToolCategory.READ_ONLY,
        description="Get calendar event details",
        required_args=["event_id"],
        sensitive=False,
    ),
    "list_events": ToolSpec(
        name="list_events",
        category=ToolCategory.READ_ONLY,
        description="List calendar events",
        required_args=[],
        sensitive=False,
    ),
    "check_balance": ToolSpec(
        name="check_balance",
        category=ToolCategory.READ_ONLY,
        description="Check bank account balance",
        required_args=[],
        sensitive=False,
    ),
    "get_balance": ToolSpec(
        name="get_balance",
        category=ToolCategory.READ_ONLY,
        description="Get bank balance",
        required_args=[],
        sensitive=False,
    ),
    "search_files": ToolSpec(
        name="search_files",
        category=ToolCategory.READ_ONLY,
        description="Search files by keyword",
        required_args=["query"],
        sensitive=False,
    ),

    # Network Read Tool (fetch_page requires SSRF / argument inspection)
    "fetch_page": ToolSpec(
        name="fetch_page",
        category=ToolCategory.NETWORK_READ,
        description="Fetch HTML content from a web URL",
        required_args=["url"],
        sensitive=False,
    ),

    # Mutating / Sensitive Tools (require explicit authorization & argument binding)
    "send_email": ToolSpec(
        name="send_email",
        category=ToolCategory.MUTATING,
        description="Send an email to a recipient",
        required_args=["to", "subject", "body"],
        sensitive=True,
    ),
    "transfer": ToolSpec(
        name="transfer",
        category=ToolCategory.MUTATING,
        description="Transfer funds to an account",
        required_args=["to_account", "amount"],
        sensitive=True,
    ),
    "bank_transfer": ToolSpec(
        name="bank_transfer",
        category=ToolCategory.MUTATING,
        description="Execute bank wire transfer",
        required_args=["to_account", "amount"],
        sensitive=True,
    ),
    "transfer_money": ToolSpec(
        name="transfer_money",
        category=ToolCategory.MUTATING,
        description="Transfer money between accounts",
        required_args=["to_account", "amount"],
        sensitive=True,
    ),
    "delete_file": ToolSpec(
        name="delete_file",
        category=ToolCategory.MUTATING,
        description="Delete a file from the filestore",
        required_args=["path"],
        sensitive=True,
    ),
    "delete_event": ToolSpec(
        name="delete_event",
        category=ToolCategory.MUTATING,
        description="Delete a calendar event",
        required_args=["event_id"],
        sensitive=True,
    ),
    "update_event": ToolSpec(
        name="update_event",
        category=ToolCategory.MUTATING,
        description="Update a calendar event",
        required_args=["event_id"],
        sensitive=True,
    ),
    "create_calendar_event": ToolSpec(
        name="create_calendar_event",
        category=ToolCategory.MUTATING,
        description="Create a calendar event",
        required_args=["title", "start_time"],
        sensitive=True,
    ),
    "execute_code": ToolSpec(
        name="execute_code",
        category=ToolCategory.MUTATING,
        description="Execute arbitrary script",
        required_args=["code"],
        sensitive=True,
    ),
    "run_command": ToolSpec(
        name="run_command",
        category=ToolCategory.MUTATING,
        description="Run shell command",
        required_args=["command"],
        sensitive=True,
    ),
}

# Known mutating prefixes for unknown tools (default-deny triggers)
MUTATING_PREFIXES: tuple[str, ...] = (
    "delete_",
    "send_",
    "transfer_",
    "update_",
    "write_",
    "execute_",
    "run_",
    "create_",
    "modify_",
    "post_",
    "drop_",
    "remove_",
    "set_",
)

# Known safe read prefixes
SAFE_READ_PREFIXES: tuple[str, ...] = (
    "get_",
    "list_",
    "read_",
    "search_",
    "view_",
    "check_",
    "lookup_",
)

# Private / reserved IP networks blocked for network tools (SSRF protection)
DISALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918 Private
    ipaddress.ip_network("172.16.0.0/12"),    # RFC1918 Private
    ipaddress.ip_network("192.168.0.0/16"),   # RFC1918 Private
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local / Cloud Metadata (AWS/GCP/Azure)
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 private
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

DISALLOWED_HOSTS: set[str] = {
    "localhost",
    "metadata.google.internal",
    "169.254.169.254",
    "instance-data",
}


def get_tool_category(tool_name: str) -> ToolCategory:
    """Return the ToolCategory for a given tool name."""
    clean_name = tool_name.lower().strip()
    if clean_name in TOOL_MANIFEST:
        return TOOL_MANIFEST[clean_name].category
    
    if any(clean_name.startswith(p) for p in MUTATING_PREFIXES):
        return ToolCategory.MUTATING
    if clean_name == "fetch_page" or "fetch" in clean_name:
        return ToolCategory.NETWORK_READ
    if any(clean_name.startswith(p) for p in SAFE_READ_PREFIXES):
        return ToolCategory.READ_ONLY
    
    return ToolCategory.UNKNOWN


def is_tool_sensitive(tool_name: str) -> bool:
    """Return True if tool is sensitive or mutating."""
    clean_name = tool_name.lower().strip()
    if clean_name in TOOL_MANIFEST:
        return TOOL_MANIFEST[clean_name].sensitive
    category = get_tool_category(clean_name)
    return category in (ToolCategory.MUTATING, ToolCategory.UNKNOWN)


def validate_network_tool_args(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    """Validate argument constraints for network-capable read tools (fetch_page).
    
    Returns:
        (is_valid, reason)
    """
    if tool_name.lower() != "fetch_page":
        return True, "Not a network tool"
    
    url = arguments.get("url") or arguments.get("link") or arguments.get("target")
    if not url:
        return False, "Missing required 'url' parameter for fetch_page"
    
    url_str = str(url).strip()
    if len(url_str) > 2048:
        return False, "URL exceeds maximum allowable length of 2048 characters"
    
    try:
        parsed = urlparse(url_str)
    except Exception as e:
        return False, f"Malformed URL: {e}"
    
    # 1. Scheme check: only http and https allowed
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Disallowed URL scheme '{parsed.scheme}'. Only 'http' and 'https' are permitted."
    
    # 2. Hostname check
    host = parsed.hostname
    if not host:
        return False, "URL does not contain a valid hostname"
    
    host_lower = host.lower()
    if host_lower in DISALLOWED_HOSTS:
        return False, f"SSRF Protection: Access to private/metadata host '{host}' is strictly blocked"
    
    # 3. IP address check (disallow loopback, RFC1918, cloud metadata)
    try:
        ip = ipaddress.ip_address(host_lower)
        for net in DISALLOWED_NETWORKS:
            if ip in net:
                return False, f"SSRF Protection: Destination IP '{ip}' is in private/restricted range '{net}'"
    except ValueError:
        # Not a raw IP literal, it is a standard domain name
        pass
    
    # 4. Port check: allow standard 80, 443, 8080, 8443
    if parsed.port and parsed.port not in (80, 443, 8080, 8443):
        return False, f"Disallowed network port: {parsed.port}. Only standard web ports (80, 443) are permitted."
    
    return True, "URL passes network security constraints"
