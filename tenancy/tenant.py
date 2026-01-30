from fastapi import Header

def get_tenant_id(x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")):
    return (x_tenant_id or "default").strip()
