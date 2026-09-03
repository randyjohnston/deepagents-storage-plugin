"""Smoke test: OffloadingBackend over a dict-backed inner + InMemoryBlobStore."""
from deepagents.backends.protocol import BackendProtocol, ReadResult, WriteResult, EditResult, DeleteResult, FileUploadResponse, FileDownloadResponse
from deepagents.backends.utils import create_file_data, update_file_data, file_data_to_string, slice_read_response, perform_string_replacement
import base64

class DictInner(BackendProtocol):
    """Minimal stand-in for StateBackend (which needs a live graph context)."""
    def __init__(self): self.files = {}
    def read(self, p, offset=0, limit=2000):
        fd = self.files.get(p)
        if fd is None: return ReadResult(error=f"File '{p}' not found")
        return slice_read_response(fd, offset, limit)
    def write(self, p, content):
        prev = self.files.get(p)
        self.files[p] = update_file_data(prev, content) if prev else create_file_data(content)
        return WriteResult(path=p)
    def edit(self, p, old, new, replace_all=False):
        fd = self.files.get(p)
        if fd is None: return EditResult(error="not found")
        r = perform_string_replacement(file_data_to_string(fd), old, new, replace_all)
        if isinstance(r, str): return EditResult(error=r)
        self.files[p] = update_file_data(fd, r[0]); return EditResult(path=p, occurrences=r[1])
    def delete(self, p):
        self.files.pop(p, None); return DeleteResult(path=p)
    def upload_files(self, files):
        out = []
        for p, data in files:
            try: self.write(p, data.decode("utf-8"))
            except UnicodeDecodeError:
                fd = create_file_data(base64.b64encode(data).decode()); fd["encoding"]="base64"; self.files[p]=fd
            out.append(FileUploadResponse(path=p))
        return out
    def download_files(self, paths):
        out = []
        for p in paths:
            fd = self.files.get(p)
            if fd is None: out.append(FileDownloadResponse(path=p, error="file_not_found")); continue
            s = file_data_to_string(fd)
            raw = base64.b64decode(s) if fd.get("encoding")=="base64" else s.encode()
            out.append(FileDownloadResponse(path=p, content=raw))
        return out

from deepagents_blob import OffloadingBackend, EXTERNAL_ENCODING, parse_pointer
from deepagents_blob.stores import InMemoryBlobStore

inner, store = DictInner(), InMemoryBlobStore()
b = OffloadingBackend(inner, store, threshold=100)

# 1. small write passes through
assert b.write("/small.txt", "hello").error is None
assert inner.files["/small.txt"]["encoding"] == "utf-8"

# 2. large write offloads: inner holds a stub, blob store holds the bytes
big = "line one\n" + ("x" * 5000) + "\nlast line"
assert b.write("/big.txt", big).error is None
stub = inner.files["/big.txt"]
assert parse_pointer(stub) is not None and len(stub["content"]) < 300, "state stays small"
ptr = parse_pointer(stub); assert ptr and store.exists(ptr["key"])

# 3. read dereferences + windows correctly
r = b.read("/big.txt", offset=0, limit=1)
assert r.error is None and r.file_data["content"] == "line one\n" and r.total_lines == 3, (r.error, r.total_lines)
r2 = b.read("/big.txt", offset=2, limit=10)
assert r2.file_data["content"] == "last line" and r2.start_line == 3

# 4. edit materializes, replaces, re-offloads
e = b.edit("/big.txt", "line one", "LINE 1")
assert e.occurrences == 1, e.error
assert "LINE 1" in b.read("/big.txt", 0, 1).file_data["content"]
assert parse_pointer(inner.files["/big.txt"]) is not None  # still offloaded

# 5. edit shrinking below threshold re-inlines
assert b.write("/shrink.txt", "y" * 200).error is None
assert parse_pointer(inner.files["/shrink.txt"]) is not None
assert b.edit("/shrink.txt", "y" * 200, "tiny").error is None
assert parse_pointer(inner.files["/shrink.txt"]) is None, "shrunk file re-inlined"

# 6. binary upload_files -> stub; download_files -> original bytes
blob = bytes(range(256)) * 10
assert b.upload_files([("/bin.dat", blob)])[0].error is None
assert parse_pointer(inner.files["/bin.dat"]) is not None
d = b.download_files(["/bin.dat", "/small.txt", "/nope"])
assert d[0].content == blob and d[1].content == b"hello" and d[2].error == "file_not_found"

# 7. content-addressing dedupes
before = len(store._objects)
b.write("/big-copy.txt", big.replace("line one", "LINE 1"))  # same content as edited /big.txt
assert len(store._objects) == before, "identical content did not re-upload"

# 8. missing blob surfaces a clean error
store._objects.clear(); b._cache = type(b._cache)(0)
assert "not found" in b.read("/big.txt").error

# 9. delete removes stub only (blob GC is lifecycle-based)
assert b.delete("/big.txt").error is None and "/big.txt" not in inner.files

print("ALL SMOKE TESTS PASSED")
