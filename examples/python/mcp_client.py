"""
MCP (Model Context Protocol) client usage with inference.rs Python API.

Connects to an MCP server, auto-discovers tools, and makes them available
to the model during conversations.

Install the filesystem server first: npx @modelcontextprotocol/server-filesystem . -y
"""

import inference_rs as ir
from inference_rs import types as t


def main():
    mcp = t.McpClientConfig(
        servers=[
            t.McpServerConfig(
                id="filesystem",
                name="Filesystem Tools",
                source=t.McpServerSourceProcess(
                    command="npx",
                    args=["@modelcontextprotocol/server-filesystem", "."],
                ),
            )
        ]
    )

    # Other transport types are also supported:
    #
    # t.McpServerSourceHttp(url="https://hf.co/mcp", timeout_secs=30)
    # t.McpServerSourceWebSocket(url="wss://api.example.com/mcp", timeout_secs=30)
    #
    # For authentication, set bearer_token="your-token" on the server config.
    # To avoid tool name conflicts, set tool_prefix="prefix".

    spec = t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="Qwen/Qwen3-4B", arch=t.NormalLoaderType.QWEN3
        ),
        agentic=t.AgenticSpec(mcp=mcp),
    )

    with ir.Engine(spec) as engine:
        response = engine.chat(
            t.ChatCompletionRequest(
                model="default",
                messages=[
                    t.Message(
                        role="user", content="List the files in the current directory."
                    )
                ],
                max_tokens=1000,
                tool_choice="auto",
            )
        )
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
