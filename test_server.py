import asyncio
from fastmcp import Client
from randoneo_support import mcp

async def main():
    async with Client(mcp) as client:
        # 1. Tools
        tools = await client.list_tools()
        print(f"=== {len(tools)} tools ===")
        for tool in tools:
            print(f"  - {tool.name} : {tool.description.splitlines()[0]}")
        
        # 2. Test d'un tool
        print("\n=== Test get_order_status ===")
        result = await client.call_tool("get_order_status", {"order_id": "RND-10238"})
        print(result.data)
        
        # 3. Test d'une resource
        print("\n=== Test product_sheet ===")
        sheet = await client.read_resource("randoneo://product/TNT-2P-AERO")
        print(sheet[0].text[:200])
        
        # 4. Test search_knowledge_base
        print("\n=== Test search_knowledge_base ===")
        docs = await client.call_tool("search_knowledge_base", {"query": "délai retour"})
        for doc in docs.data[:2]:
            print(f"  - {doc[:80]}...")

asyncio.run(main())
