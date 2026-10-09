"""
Agent Randoneo — sans aucun outil en dur.

Les capacités viennent du serveur MCP (randoneo_support.py).
Le garde-fou humain est géré dans chat.py.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Charge le .env
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_PROMPT = (
    "Tu es l'assistant du support client de Randoneo, un vendeur de matériel "
    "de randonnée. Utilise tes outils pour renseigner le client. "
    "Réponds en français, de façon concise et professionnelle."
)

client = MultiServerMCPClient({
    "randoneo": {
        "command": "python",
        "args": ["randoneo_support.py"],
        "transport": "stdio",
        "env": {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "PATH": os.environ["PATH"],
        },
    }
})


# ============================================================
# CONSTRUCTION DE L'AGENT
# ============================================================

async def build_agent(checkpointer):
    """Charge les tools du serveur MCP et construit l'agent."""
    mcp_tools = await client.get_tools()
    print(f"✅ {len(mcp_tools)} tools chargés depuis le MCP")
    for tool_item in mcp_tools:
        print(f"   - {tool_item.name}")
    
    model = init_chat_model(
        "claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0,
        max_retries=8,
    )
    
    agent = create_agent(
        model,
        mcp_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return agent


# ============================================================
# TEST
# ============================================================

async def main():
    DB = "randoneo_memory.sqlite"
    
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        agent = await build_agent(checkpointer)
        
        print("\n" + "=" * 60)
        print("Test de l'agent")
        print("=" * 60 + "\n")
        
        config = {"configurable": {"thread_id": "test-1"}}
        
        # Test 1
        print("❓ Test 1 : Où en est ma commande RND-10235 ?")
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Où en est ma commande RND-10235 ?")]},
            config,
        )
        print(f"💬 {result['messages'][-1].content}\n")
        
        # Test 2
        print("❓ Test 2 : Et elle arrive quand ?")
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Et elle arrive quand ?")]},
            config,
        )
        print(f"💬 {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())