# tools/desktop/approvals.py
import json, os, uuid, datetime

FILE = "desktop_approvals.json"

def _load():
    if not os.path.exists(FILE):
        return []
    try:
        return json.load(open(FILE))
    except Exception:
        return []

def _save(arr):
    with open(FILE, "w") as f:
        json.dump(arr, f, indent=2, default=str)

def add_request(action, args, requested_by):
    arr = _load()
    req = {
        "id": str(uuid.uuid4()),
        "action": action,
        "args": args,
        "requested_by": requested_by,
        "status": "pending",
        "ts": datetime.datetime.utcnow().isoformat()
    }
    arr.append(req)
    _save(arr)
    return req

def set_status(req_id, status):
    arr = _load()
    for r in arr:
        if r["id"] == req_id:
            r["status"] = status
            r["updated_ts"] = datetime.datetime.utcnow().isoformat()
    _save(arr)

def find_request(req_id):
    for r in _load():
        if r["id"] == req_id:
            return r
    return None
