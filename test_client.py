import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1. Define server parameters
    server_params = StdioServerParameters(
        command = "python3",
        args = ["server.py"]
    )

    print("---  Connecting to ELECTION MCP SERVER ---")
    
    # 2. Establish connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # --- TEST 1: LIST TOOLS ---
            tools = await session.list_tools()
            print(f"\n Server Connected! Found {len(tools.tools)} tools.")
            for t in tools.tools:
                print(f"   - {t.name} : {t.description}")
            
            # --- TEST 2: GET SENTIMENT ---
            print("\n Requesting sentiment for Thiruvananthapuram...")
            response = await session.call_tool("get_election_sentiment", {
                "state": "Kerala",
                "constituency": "Thiruvananthapuram"
            })

            print("\n Tool response:")
            sentiment_text = response.content[0].text
            print(sentiment_text)

            # --- TEST 3: POLITICAL RISK ANALYSIS ---
            print("\n Requesting risk analysis for the result...")
            # Parse the string result back into a dictionary
            gemini_dict = json.loads(sentiment_text)
            
            risk_response = await session.call_tool("get_political_risk_analysis", {
                "gemini_output": gemini_dict
            })
            
            print("\n Risk Analysis Response:")
            print(risk_response.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())