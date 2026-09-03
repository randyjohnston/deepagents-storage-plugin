"""End-to-end: a real deepagents graph whose filesystem tools flow through
OffloadingBackend, driven by a scripted tool-calling model (no API key).

Proves the full integration the smoke test can't: create_deep_agent wiring,
the write_file/read_file tools, and the files channel in the final state.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "unused-fake-model")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemState

from deepagents_blob import OffloadingBackend, parse_pointer
from deepagents_blob.stores import InMemoryBlobStore

THRESHOLD = 1_000
BIG = "header,value\n" + "\n".join(f"row-{i},{i % 7}" for i in range(2_000))
assert len(BIG.encode()) > THRESHOLD


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that ignores bound tools and replays a script."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


script = iter(
    [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/data/report.csv", "content": BIG},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/data/report.csv", "offset": 0, "limit": 3},
                    "id": "call_2",
                }
            ],
        ),
        AIMessage(content="Saved /data/report.csv and verified its header."),
    ]
)

store = InMemoryBlobStore()
backend = OffloadingBackend(StateBackend(), store, threshold=THRESHOLD)
# state_schema=FilesystemState is required: deepagents' _uses_state_backend only
# recognizes StateBackend/CompositeBackend, so a decorator backend must register
# the `files` channel itself.
agent = create_deep_agent(model=ScriptedModel(messages=script), backend=backend, state_schema=FilesystemState)

result = agent.invoke({"messages": [{"role": "user", "content": "Save the report."}]})

# 1. The files channel holds a pointer stub, not the 30KB payload.
stub = result["files"]["/data/report.csv"]
ptr = parse_pointer(stub)
assert ptr is not None, "large write was not offloaded"
assert len(stub["content"]) < 300, "state should hold only the stub"
assert ptr["size"] == len(BIG.encode())

# 2. The bytes live in the blob store under the content-addressed key.
assert store.exists(ptr["key"])
assert store.get(ptr["key"]).decode() == BIG

# 3. The read_file tool dereferenced the stub for the model.
tool_msgs = [m for m in result["messages"] if m.type == "tool" and m.tool_call_id == "call_2"]
assert tool_msgs and "header,value" in str(tool_msgs[0].content), "read_file did not materialize blob"

print("E2E PASSED: state holds a", len(stub["content"]), "byte stub;", ptr["size"], "bytes offloaded to", ptr["key"])
